import numpy as np 
paloma_path = "/home/user/data/paloma/tokenized_paloma_c4_100_domains_validation.bin"

# Read the binary data
data = np.fromfile(
    paloma_path,
    dtype=np.uint16
)

num_samples = 2000

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2") 

# Filtering imports
from cs336_data.quality_filter import passes_quality_filters
from cs336_data.language_identifier import find_language_scores
from cs336_data.nsfw_toxic import find_nsfw, find_toxic

# Decode the entire data
text = tokenizer.decode(data)

# Split by the special endoftext token
endoftext_token = "<|endoftext|>"
documents = text.split(endoftext_token)

positive_docs = 0
rejected_docs = {
    "failed language": 0,
    "failed nsfw": 0,
    "failed toxic": 0,
    "failed basic quality": 0,
}

from tqdm import tqdm
for doc in tqdm(documents[0:num_samples], desc="Filtering", unit="doc"):
    doc = doc.strip()
    # if not doc:
    #     continue
    # if find_language_scores(doc)[0] != 'en':
    #     rejected_docs["failed language"] += 1
    #     continue
    # if not passes_quality_filters(doc):
    #     rejected_docs["failed basic quality"] += 1
    #     continue
    # if find_nsfw(doc)[0] == 'nsfw':
    #     rejected_docs["failed nsfw"] += 1
    #     continue
    # if find_toxic(doc)[0] == 'toxic':
    #     rejected_docs["failed toxic"] += 1
    #     continue
    doc = doc.replace('\n', ' ')
    # Write to file in real time
    with open("positives_paloma.txt", "a", encoding="utf-8") as f:
        print(doc)
        print("-"*100)
        f.write(doc + "<|endoftext|>\n")
        positive_docs += 1

print(f"Wrote {positive_docs} positive samples to positives_paloma.txt")
print(f"Rejected docs: {rejected_docs}")


