import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import sys


input_path = "/home/user/cs336-a4-data/filtered_outputs/filtered_example.txt"  
output_path = "/home/user/cs336-a4-data/tokenized_outputs/tokenized_example.npy"  

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_line_and_add_eos(line):
    return tokenizer.encode(line) + [tokenizer.eos_token_id]

if __name__ == "__main__":
    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    chunksize = 100
    results = []
    for result in tqdm(
        pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
        total=len(lines),
        desc="Tokenizing lines"
    ):
        results.append(result)
    pool.close()
    pool.join()

  
    all_ids = [token_id for sublist in results for token_id in sublist]
    print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)
