import os
import time

import multiprocessing as mp
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.text_extractor import extract_text_from_warc
from cs336_data.text_extractor import extract_text            # byte‑>plain‑text
from cs336_data.quality_filter import passes_quality_filters  # quick heuristics
from cs336_data.language_identifier import find_language_scores
from cs336_data.personal_mask import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.nsfw_toxic import find_nsfw, find_toxic
from cs336_data.dedup import compute_hash   


input_file = "/home/user/cs336-a4-data/datagen/positive_urls.warc.gz"
output_file = "/home/user/cs336-a4-data/datagen/filtered_positive_urls.warc.gz"

num_samples = 12000
def read_docs(input_file, max_records=None):
    """Read documents from either a WARC or plain text file. For .txt, split on <|endoftext|>."""
    docs = []
    if (
        input_file.endswith('.warc.gz')
        or input_file.endswith('.warc')
        or input_file.endswith('.warc.wet.gz')
    ):
        docs = extract_text_from_warc(input_file, max_records=max_records)
    elif input_file.endswith('.txt'):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
            docs = [doc.strip() for doc in text.split('<|endoftext|>') if doc.strip()]
            if max_records is not None:
                docs = docs[:max_records]
    else:
        raise ValueError(f"Unsupported file type for input_file: {input_file}")
    return docs

def make_positive(input_file, output_file):

    rejected_docs = {
        "failed language": 0,
        "failed nsfw": 0,
        "failed toxic": 0,
        "failed basic quality": 0,
    }

    positive_docs = []
    num_processed = 0

    docs = read_docs(input_file, max_records=10000)
    # total_docs = len(docs)
    # start_time = time.time()
    # for text in docs:

    #     if find_language_scores(text)[0] != 'en':
    #         rejected_docs["failed language"] += 1
    #         print(f"Failed language: found {find_language_scores(text)[0]} in {text}")

    #     elif not passes_quality_filters(text):
    #         rejected_docs["failed basic quality"] += 1
    #         print(f"Failed basic quality: {text}")

    #     elif find_nsfw(text) == 'nsfw':
    #         rejected_docs["failed nsfw"] += 1
    #         print(f"Failed nsfw: {text}")   

    #     elif find_toxic(text) == 'toxic':
    #         rejected_docs["failed toxic"] += 1
    #         print(f"Failed toxic: {text}")

    #     else:
    #         text = text.replace('\n', ' ')
    #         positive_docs.append(text)
            

    #     num_processed += 1
    #     if num_processed % 50 == 0:
    #         elapsed = time.time() - start_time
    #         rate = num_processed / elapsed if elapsed > 0 else 0
    #         remaining = total_docs - num_processed
    #         eta = remaining / rate if rate > 0 else 0
    #         print(f"Processed {num_processed} docs")
    #         print(f"Passed docs: {len(positive_docs)}")
    #         print(f"ETA: {eta:.1f} seconds left")

    # print(f"Processed {num_processed} docs")
    # print(f"Positive docs: {len(positive_docs)}")
    # print(f"Rejected docs: {rejected_docs}")
    positive_docs = docs
    # for doc in positive_docs:
    #     print(doc)
    #     print("-"*100)
    print(f"here!!!")
    return positive_docs #, rejected_docs

negative_input_file = "/home/user/data/CC/CC-MAIN-20250421065628-20250421095628-00882.warc.wet.gz"
def make_negative(negative_input_file, output_file):
    negative_docs = []
    print(f"Reading negative docs from {negative_input_file}")
    docs = read_docs(negative_input_file, max_records=num_samples)
    for text in docs:
        text = text.replace('\n', ' ')

        if find_language_scores(text)[0] == 'en':
            negative_docs.append(text)
        if len(negative_docs) % 100 == 0:
            print(f"Negative docs: {len(negative_docs)}")
            # print(f"ETA: {eta:.1f} seconds left")

    print(f"Negative docs: {len(negative_docs)}")
    return negative_docs
    
train_path = "/home/user/cs336-a4-data/datagen/train.txt"
valid_path = "/home/user/cs336-a4-data/datagen/valid.txt"
valid_ratio = 0.1

def write_samples(samples, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.strip() + "\n")


def make_dataset(positive_input_file, negative_input_file, combined_out):
    positive_docs = make_positive(positive_input_file, None)
    print(f"here!!!2")
    negative_docs = make_negative(negative_input_file, None)
    write_samples(positive_docs, "positives.txt")
    write_samples(negative_docs, "negatives.txt")
    # Combine and label: 1 for positive, 0 for negative, using fastText format
    all_docs = [(doc, 1) for doc in positive_docs] + [(doc, 0) for doc in negative_docs]
    with open(combined_out, "w", encoding="utf-8") as f:
        for doc, label in all_docs:
            doc = doc.replace('\n', ' ')
            f.write(f"__label__{label} {doc.strip()}\n")


def balanced_valid_train_split(input_file, valid_path, train_path, valid_ratio=0.1):
    # Read and separate by label (fastText format)
    print(f"generating balanced valid train split")
    positives = []
    negatives = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("__label__"):
                continue  # skip empty or malformed lines
            if line.startswith("__label__1 "):
                positives.append(line + "\n")
            elif line.startswith("__label__0 "):
                negatives.append(line + "\n")
    # Compute split sizes
    valid_size_pos = int(len(positives) * valid_ratio)
    valid_size_neg = int(len(negatives) * valid_ratio)
    # No shuffle for simplicity, but you can add random.shuffle(positives) etc. if you want
    train_lines = positives[:-valid_size_pos] + negatives[:-valid_size_neg]
    valid_lines = positives[-valid_size_pos:] + negatives[-valid_size_neg:]
    # Optionally shuffle train/valid lines here for randomness
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(valid_path, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)


# Example usage:
if __name__ == "__main__":
    positive_input = "/home/user/cs336-a4-data/positives_paloma.txt"
    negative_input = negative_input_file
    combined_out = "/home/user/cs336-a4-data/datagen/all_samples.txt"
    train_path = "/home/user/cs336-a4-data/datagen/train.txt"
    valid_path = "/home/user/cs336-a4-data/datagen/valid.txt"

    make_dataset(positive_input, negative_input, combined_out)
    balanced_valid_train_split(combined_out, valid_path, train_path, valid_ratio=0.1)







        

       









