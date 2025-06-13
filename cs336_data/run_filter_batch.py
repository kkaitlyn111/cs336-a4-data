from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys
import time
import multiprocessing as mp
from fastwarc.warc import ArchiveIterator, WarcRecordType

from tqdm import tqdm

# ---------------------------------------------------------------------------
# import the previously‑written pipeline
# ---------------------------------------------------------------------------

from cs336_data.filter_CC import process_single_wet_file, gather_all_stats

# ---------------------------------------------------------------------------
MAX_DOCS = 10000

def count_documents_in_file(wet_path):
    count = 0
    with open(wet_path, "rb") as f:
        for _ in ArchiveIterator(f, record_types=WarcRecordType.conversion):
            count += 1
    return count

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Parallel CommonCrawl filtering")
    p.add_argument("--input-root", default="/home/user/data/CC_example",
                   help="Directory with CC*.warc.wet.gz files")
    p.add_argument("--output-root", default="/home/user/cs336-a4-data/filtered_outputs",
                   help="Output directory for filtered shards")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of worker processes (default = CPU cores)")
    p.add_argument("--start-doc", type=int, default=1,
                   help="Start document index (1-based, inclusive)")
    p.add_argument("--end-doc", type=int, default=MAX_DOCS,
                   help="End document index (1-based, inclusive). If None, process to end.")
    args = p.parse_args(argv)

    in_dir = pathlib.Path(args.input_root)
    out_dir = pathlib.Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    wet_files = sorted(in_dir.glob("*.warc.wet.gz"))
    if not wet_files:
        print(f"[!] No shards found under {in_dir}", file=sys.stderr)
        sys.exit(1)

    print("[+] Counting total documents in all shards...")
    total_docs = 0
    for fp in tqdm(wet_files, desc="Counting docs", unit="file"):
        total_docs += count_documents_in_file(fp)
    print(f"[+] Total documents to process: {total_docs}")

    n_workers = args.workers or os.cpu_count() or 1
    print(f"[+] Filtering {len(wet_files)} shards using {n_workers} workers …")

    manager = mp.Manager()
    doc_counter = manager.Value('i', 0)

    stats_list: list[dict[str, int]] = []

    def update_progress_bar(pbar, doc_counter, total_docs, start_time):
        last_count = 0
        while pbar.n < total_docs:
            current = doc_counter.value
            pbar.n = current
            pbar.refresh()
            elapsed = time.time() - start_time
            throughput = current / elapsed if elapsed > 0 else 0
            eta = (total_docs - current) / throughput if throughput > 0 else float('inf')
            pbar.set_postfix({
                'throughput': f"{throughput:.2f} it/s",
                'eta': f"{eta:.1f} s"
            })
            if current >= total_docs:
                break
            time.sleep(0.5)
        pbar.n = total_docs
        pbar.refresh()

    # Start progress bar thread
    from threading import Thread
    from tqdm import tqdm as tqdm_cls
    start_time = time.time()
    pbar = tqdm_cls(total=total_docs, desc="documents", unit="doc")
    progress_thread = Thread(target=update_progress_bar, args=(pbar, doc_counter, total_docs, start_time))
    progress_thread.start()

    with cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_single_wet_file, fp, out_dir, doc_counter, args.start_doc, args.end_doc): fp
                   for fp in wet_files}
        for fut in cf.as_completed(futures):
            stats_list.append(fut.result())

    progress_thread.join()
    pbar.close()

    # Aggregate + save
    agg = gather_all_stats(stats_list)
    with open(out_dir / "aggregate_stats.json", "w") as f:
        json.dump(agg, f, indent=2)

    print("\n[+] Done. Aggregate stats:")
    for k, v in agg.items():
        print(f"{k:20s}: {v:,}")


if __name__ == "__main__":
    main()
