


max_num_urls = 100000
input_path ="/home/user/data/wiki/enwiki-20240420-extracted_urls.txt.gz"
output_path = "/home/user/cs336-a4-data/datagen/positive_urls.txt"


import gzip

n = max_num_urls

# Read first n URLs and write to output
with gzip.open(input_path, 'rt') as fin, open(output_path, 'w') as fout:
    for i, line in enumerate(fin):
        if i >= n:
            break
        if line.strip():
            fout.write(line)  

print(f"Wrote {min(i+1, n)} URLs to {output_path}")






