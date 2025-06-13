from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys
from collections import Counter

from fastwarc.stream_io import GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

# ---------------------------------------------------------------------------
# local helpers – we intentionally import *after* adding /mnt/data to path
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# --- core primitives implemented by you ------------------------------------
from text_extractor import extract_text            # byte‑>plain‑text
from quality_filter import passes_quality_filters  # quick heuristics
# from quality_classifier_better import classify_quality
from language_identifier import find_language_scores
from personal_mask import mask_emails, mask_phone_numbers, mask_ips
from nsfw_toxic import find_nsfw, find_toxic
from dedup import compute_hash                      # exact intra‑run dedup

# ----------------------------------------------------------------------------
FILTER_ORDER = [
    "language",
    "nsfw",
    "toxic",
    "quality_basic",
    "quality_classifier",
]

# Boolean helper – nicer than repeatedly writing [0] == 'label'

def _is(label_pred, target: str) -> bool:
    return label_pred[0] == target


def mask_pii(text: str) -> str:
    """Return *lower‑cased* text with PII replaced by sentinel tokens."""
    text = text.lower()
    text, _ = mask_emails(text)
    text, _ = mask_phone_numbers(text)
    text, _ = mask_ips(text)
    return text


# ---------------------------------------------------------------------------

def filter_document(text: str, counters: Counter, doc_idx: int = None, doc_hash: str = None) -> bool:
    """Return *True* if the document should be retained, updating `counters`. Optionally logs doc index/hash."""
    counters["total"] += 1

    # --- step 0: PII masking (always) --------------------------------------
    text = mask_pii(text)
    if doc_idx is not None:
        print(f"[Doc {doc_idx}] Hash: {doc_hash if doc_hash else 'N/A'} - After PII masking.")

    # --- 1. Language -------------------------------------------------------
    lang, _score = find_language_scores(text)
    if lang != "en":
        counters["language"] += 1
        print(f"[Doc {doc_idx}] Filtered: Not English (lang={lang})")
        return False

    # --- 2. NSFW -----------------------------------------------------------
    if _is(find_nsfw(text), "nsfw"):
        counters["nsfw"] += 1
        print(f"[Doc {doc_idx}] Filtered: NSFW")
        return False

    # --- 3. Hate / toxicity -----------------------------------------------
    if _is(find_toxic(text), "toxic"):
        counters["toxic"] += 1
        print(f"[Doc {doc_idx}] Filtered: Toxic")
        return False

    # --- 4. Quick heuristics ----------------------------------------------
    if not passes_quality_filters(text):
        counters["quality_basic"] += 1
        print(f"[Doc {doc_idx}] Filtered: Failed quality heuristics")
        return False

    # # --- 5. fastText quality classifier -----------------------------------
    # label, _prob = classify_quality(text)
    # # The classifier converts __label__wiki → "wiki", __label__cc → "cc"
    # if label != "wiki":
    #     counters["quality_classifier"] += 1
    #     print(f"[Doc {doc_idx}] Filtered: Quality classifier label={label}")
    #     return False

    # ----------------------------------------------------------------------
    counters["kept"] += 1
    print(f"[Doc {doc_idx}] Kept.")
    return True


# ---------------------------------------------------------------------------

def process_single_wet_file(wet_path: pathlib.Path, out_dir: pathlib.Path, doc_counter=None, start_doc: int = 1, end_doc: int = None) -> dict[str, int]:
    """Filter a single .warc.wet.gz file and return the *local* counters. Logs progress. Only processes documents in [start_doc, end_doc] (1-based, inclusive)."""
    counters: Counter = Counter()
    import tempfile
    from minhashLSHdedup import run_minhash_deduplication

    out_file = out_dir / (wet_path.stem + ".txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = tempfile.TemporaryDirectory()
    temp_files = []
    doc_texts = []

    # Streaming read (decompress on the fly)
    with GZipStream(open(wet_path, "rb")) as gz_stream:
        print(f"[+] Processing file: {wet_path}")
        for doc_idx, record in enumerate(ArchiveIterator(gz_stream, record_types=WarcRecordType.conversion), 1):
            if doc_idx < start_doc:
                if doc_counter is not None:
                    doc_counter.value += 1
                continue
            if end_doc is not None and doc_idx > end_doc:
                break
            try:
                raw_bytes = record.reader.read()
                plain = extract_text(raw_bytes)
                if not plain:
                    print(f"[Doc {doc_idx}] Skipped: No text extracted.")
                    if doc_counter is not None:
                        doc_counter.value += 1
                    continue
            except Exception as e:
                counters["decode_error"] += 1
                print(f"[Doc {doc_idx}] Skipped: Decode error: {e}")
                if doc_counter is not None:
                    doc_counter.value += 1
                continue

            # Compute hash before filtering for logging
            h = compute_hash(plain)

            if not filter_document(plain, counters, doc_idx=doc_idx, doc_hash=h):
                if doc_counter is not None:
                    doc_counter.value += 1
                continue

            # Instead of exact dedup, collect filtered docs as temp files for LSH dedup
            temp_path = pathlib.Path(temp_dir.name) / f"doc_{doc_idx}.txt"
            with open(temp_path, "w", encoding="utf-8") as tf:
                tf.write(plain)
            temp_files.append(str(temp_path))
            doc_texts.append(plain)
            if doc_counter is not None:
                doc_counter.value += 1

    # Run MinHash LSH deduplication on temp files
    # Default parameters: n=5, k=100, bands=20, threshold=0.8
    dedup_dir = pathlib.Path(temp_dir.name) / "deduped"
    run_minhash_deduplication(temp_files, n=5, k=100, bands=20, threshold=0.8, output_dir=dedup_dir)

    # Write deduplicated docs to output file
    with open(out_file, "w", encoding="utf-8") as fout:
        for deduped_file in sorted(dedup_dir.glob("*.txt")):
            with open(deduped_file, "r", encoding="utf-8") as df:
                doc = df.read()
                fout.write(doc.replace("\n", " ").strip() + " <|endoftext|>\n")
                print(f"[Deduped] Written to output. File: {deduped_file.name}")
                counters["kept"] += 1

    # Clean up temp files
    temp_dir.cleanup()

    # write per‑file stats next to the shard (JSON)
    stats_path = out_file.with_suffix(".stats.json")
    with open(stats_path, "w") as statsf:
        json.dump(counters, statsf, indent=2)

    print(f"[+] Finished processing {wet_path}. Stats: {dict(counters)}")
    return dict(counters)


# ---------------------------------------------------------------------------

def gather_all_stats(stats_list: list[dict[str, int]]) -> Counter:
    agg: Counter = Counter()
    for stats in stats_list:
        agg.update(stats)
    return agg


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Filter Common Crawl data.")
    parser.add_argument("--input-root", default="/data/CC", help="Directory containing CC*.warc.wet.gz shards")
    parser.add_argument("--output-root", default="/data/CC_filtered", help="Where the filtered .txt shards are written")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (processes). 0 = serial")
    args = parser.parse_args(argv)

    in_dir = pathlib.Path(args.input_root)
    out_dir = pathlib.Path(args.output_root)
    wet_files = sorted(in_dir.glob("CC*.warc.wet.gz"))

    if not wet_files:
        print(f"[!] No WET shards found under {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Found {len(wet_files)} input shards; writing to {out_dir}")

    stats_list: list[dict[str, int]] = []

    if args.workers and args.workers > 1:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_single_wet_file, fp, out_dir): fp for fp in wet_files}
            for fut in tqdm(cf.as_completed(futures), total=len(wet_files), desc="shards"):
                stats_list.append(fut.result())
    else:
        for fp in tqdm(wet_files, desc="shards"):
            stats_list.append(process_single_wet_file(fp, out_dir))

    # aggregate
    agg = gather_all_stats(stats_list)
    out_stats = out_dir / "aggregate_stats.json"
    with open(out_stats, "w") as f:
        json.dump(agg, f, indent=2)

    print("\n[+] Aggregate statistics:")
    for key, val in agg.items():
        print(f"{key:20s} : {val:,}")

    # Optional: exact *cross‑shard* dedup — reuse dedup.exact_deduplication ----
    # Uncomment this block if you want to deduplicate across shards after filtering.
    # from dedup import exact_deduplication
    # print("[+] Running exact deduplication across shards…")
    # shard_paths = [p for p in out_dir.glob("*.txt")]
    # dedup_out = out_dir / "dedup"
    # dedup_out.mkdir(exist_ok=True)
    # exact_deduplication(shard_paths, dedup_out)
    # print("[+] Deduplicated files written to", dedup_out)


if __name__ == "__main__":
    main()
