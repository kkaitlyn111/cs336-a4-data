import fasttext
from cs336_data.text_extractor import extract_text_from_warc
from cs336_data.quality_filter import passes_quality_filters
from typing import Iterator, Tuple
import random
import gzip
import os

# 1. Subsample Wikipedia URLs
def subsample_wiki_urls(wiki_url_gz: str, output_file: str, n: int = 1000, seed: int = 42):
    random.seed(seed)
    with gzip.open(wiki_url_gz, 'rt') as f:
        urls = [line.strip() for line in f if line.strip()]
    sampled = random.sample(urls, min(n, len(urls)))
    with open(output_file, 'w') as out:
        for url in sampled:
            out.write(url + '\n')
    print(f"Wrote {len(sampled)} URLs to {output_file}")

# 2. Extract text from WARC to plain text file
def extract_texts_to_file(warc_file: str, output_file: str, max_records: int = 10000):
    docs = extract_text_from_warc(warc_file, max_records=max_records)
    with open(output_file, 'w') as f:
        for doc in docs:
            f.write(doc.replace('\n', ' ') + '\n')
    print(f"Extracted {len(docs)} documents to {output_file}")

# 3. Filter documents using quality filter
def filter_documents(input_file: str, output_file: str):
    count_in, count_out = 0, 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            count_in += 1
            text = line.strip()
            if passes_quality_filters(text):
                fout.write(text + '\n')
                count_out += 1
    print(f"Filtered {count_in} -> {count_out} high-quality documents.")

# 4. Write fastText training file
def write_training_file(positive_file: str, negative_warc: str, output_file: str, max_records: int = 10000):
    with open(output_file, 'w') as f:
        # Positives
        with open(positive_file, 'r') as pos:
            for line in pos:
                f.write(f"__label__high {line.strip()}\n")
        # Negatives
        for text in extract_text_from_warc(negative_warc, max_records=max_records):
            f.write(f"__label__low {text.replace('\n', ' ')}\n")
    print(f"Wrote training file to {output_file}")

# 5. Train fastText classifier
def train_fasttext_classifier(train_file, model_file):
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.5,
        epoch=10,
        wordNgrams=2,
        verbose=2,
        minCount=5
    )
    model.save_model(model_file)
    print(f"Model saved to {model_file}")
    return model

# 6. Adapter function for classifier
def run_classify_quality(text: str) -> Tuple[str, float]:
    global _quality_model
    if '_quality_model' not in globals():
        _quality_model = fasttext.load_model('quality_classifier.ftz')
    labels, probs = _quality_model.predict(text)
    label = 'high-quality' if labels[0] == '__label__high' else 'low-quality'
    confidence = float(probs[0])
    return label, confidence

def debug_filter_examples(input_file: str, output_file: str, n: int = 5):
    """Prints examples of filtered-out and passed documents for debugging."""
    passed = set()
    with open(output_file, 'r') as f:
        for line in f:
            passed.add(line.strip())
    failed = []
    shown_passed = 0
    shown_failed = 0
    print("\n--- Debug: Examples of PASSED documents ---")
    with open(input_file, 'r') as f:
        for line in f:
            text = line.strip()
            if text in passed and shown_passed < n:
                print(f"[PASSED] {text[:200]}...\n")
                shown_passed += 1
            elif text not in passed and shown_failed < n:
                failed.append(text)
                shown_failed += 1
            if shown_passed >= n and shown_failed >= n:
                break
    print("\n--- Debug: Examples of FILTERED OUT documents ---")
    for text in failed:
        print(f"[FILTERED OUT] {text[:200]}...\n")

if __name__ == "__main__":
    # # Step 1: Subsample 1000 URLs from Wikipedia links
    # wiki_url_gz = "/home/user/data/wiki/enwiki-20240420-extracted_urls.txt.gz"
    # subsampled_urls_file = "subsampled_positive_urls.txt"
    # print("Step 1: Sampled 1000 URLs from wiki links:")
    # subsample_wiki_urls(wiki_url_gz, subsampled_urls_file, n=1000)

    # print("\nStep 2: Download the positive examples in WARC format using:")
    # print(f"wget --timeout=5 -i {subsampled_urls_file} --warc-file=subsampled_positive_urls.warc -O /dev/null")
    # print("After download, decompress the .warc.wet.gz if needed.")

    # Step 3: Extract text from the WARC file
    positive_warc = "/home/user/cs336-a4-data/subsampled_positive_urls.warc.warc.gz"  # or .warc.wet if not gzipped
    positive_docs_file = "positive_documents.txt"
    if os.path.exists(positive_warc):
        extract_texts_to_file(positive_warc, positive_docs_file, max_records=10000)
    else:
        print(f"Please ensure {positive_warc} exists before continuing.")

    # Step 4: Filter positive documents
    filtered_positive_file = "filtered_positive_documents.txt"
    if os.path.exists(positive_docs_file):
        filter_documents(positive_docs_file, filtered_positive_file)
        debug_filter_examples(positive_docs_file, filtered_positive_file, n=5)
    else:
        print(f"Please ensure {positive_docs_file} exists before continuing.")

    # Step 5: Prepare training file and train classifier
    negative_warc = "/home/user/cs336-a4-data/filterdataset/negative_examples.warc.wet.gz"
    train_file = "train.txt"
    model_file = "quality_classifier.ftz"
    if os.path.exists(filtered_positive_file) and os.path.exists(negative_warc):
        write_training_file(filtered_positive_file, negative_warc, train_file)
        train_fasttext_classifier(train_file, model_file)
    else:
        print(f"Please ensure {filtered_positive_file} and {negative_warc} exist before continuing.")

    print("\nYou can now use run_classify_quality(text) to classify new documents.")