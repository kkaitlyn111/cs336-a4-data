import os
import random
import mmh3
import re
import string
import unicodedata
import shutil

# texzt normalization: lowercase, remove punctuation, normalize spaces, remove accents
def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(char for char in text if not unicodedata.combining(char))
    return unicodedata.normalize('NFD', text)

# extract char-level n-grams from file after norm
def extract_ngrams(filepath, n):
    with open(filepath, 'r') as f:
        content = normalize(f.read())
    return [content[i:i+n] for i in range(len(content) - n)]

# compute MinHash signature using k hash functions
def compute_minhash(filepath, n, k):
    ngrams = extract_ngrams(filepath, n)
    signature = []
    for seed in range(k):
        hashed = [mmh3.hash(ng, seed) for ng in ngrams]
        signature.append(min(hashed))
    return signature

# apply Locality Sensitive Hashing LSH to detect candidate duplicates
def lsh_match(file1, file2, n, k, bands):
    sig1 = compute_minhash(file1, n, k)
    sig2 = compute_minhash(file2, n, k)
    rows_per_band = k // bands
    for i in range(0, k, rows_per_band):
        if sig1[i:i+rows_per_band] == sig2[i:i+rows_per_band]:
            return True
    return False

# calc Jaccard similarity between two documents based on n-grams
def jaccard(file1, file2, n):
    ngrams1 = set(extract_ngrams(file1, n))
    ngrams2 = set(extract_ngrams(file2, n))
    if not ngrams1 and not ngrams2:
        return 1.0  # Handle empty files
    return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)


def is_duplicate(file1, file2, n, k, bands, threshold):
    return lsh_match(file1, file2, n, k, bands) and jaccard(file1, file2, n) > threshold

# cluster documents based on candidate duplicates
def cluster_documents(filepaths, n, k, bands, threshold):
    clusters = {}
    cluster_id = 0

    for idx, file_a in enumerate(filepaths):
        assigned = False
        for j in range(idx):
            file_b = filepaths[j]
            if is_duplicate(file_a, file_b, n, k, bands, threshold):
                clusters[file_a] = clusters[file_b]
                assigned = True
                break

        if not assigned:
            clusters[file_a] = cluster_id
            cluster_id += 1

    return clusters


def run_minhash_deduplication(filepaths, n, k, bands, threshold, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    clusters = cluster_documents(filepaths, n, k, bands, threshold)

    cluster_groups = {}
    for file, label in clusters.items():
        cluster_groups.setdefault(label, []).append(file)

    for group in cluster_groups.values():
        selected_file = random.choice(group)
        destination = os.path.join(output_dir, os.path.basename(selected_file))
        shutil.copy2(selected_file, destination)