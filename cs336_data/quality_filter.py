import re
from typing import List
from cs336_data.text_extractor import extract_text_from_warc
import random

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    from nltk.tokenize import word_tokenize
    def tokenize(text: str) -> List[str]:
        return word_tokenize(text)
except (ImportError, LookupError):
    def tokenize(text: str) -> List[str]:
        # Fallback: split on whitespace
        return text.split()

def passes_quality_filters(text: str) -> bool:
    words = tokenize(text)
    num_words = len(words)
    if num_words < 50 or num_words > 100_000:
        return False

    # Mean word length
    word_lengths = [len(word) for word in words]
    mean_word_length = sum(word_lengths) / num_words if num_words > 0 else 0
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # Percentage of lines ending with ellipsis
    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith('...'))
        ellipsis_ratio = ellipsis_lines / len(lines)
        if ellipsis_ratio > 0.3:
            return False

    # Percentage of words with at least one alphabetic character
    alpha_words = sum(1 for word in words if re.search(r'[A-Za-z]', word))
    if num_words == 0 or (alpha_words / num_words) < 0.8:
        return False

    return True

def main():
    warc_path = "/home/user/data/CC_example/example.warc.wet.gz"
    docs = extract_text_from_warc(warc_path, max_records=200)
    random.shuffle(docs)
    for i, doc in enumerate(docs[:20], 1):
        passed = passes_quality_filters(doc)
        print(f"--- Example {i} ---\n{doc}\n[Quality Filter: {'PASS' if passed else 'FAIL'}]\n")

if __name__ == "__main__":
    main()
