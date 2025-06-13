import hashlib
import os


def compute_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# first pass: count occurrences of each line across all files
def count_lines_in_file(filepath, line_counter):
    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            line_hash = compute_hash(line)
            line_counter[line_hash] = line_counter.get(line_hash, 0) + 1

# second pass: write only unique lines to new file
def write_unique_lines(filepath, line_counter, output_directory):
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_directory, filename)
    
    with open(filepath, 'r') as infile, open(output_path, 'w') as outfile:
        for raw_line in infile:
            line = raw_line.strip()
            line_hash = compute_hash(line)
            if line_counter[line_hash] == 1:
                outfile.write(f"{line}\n")


def exact_deduplication(file_list, output_directory):
    line_counter = {}
    

    for filepath in file_list:
        count_lines_in_file(filepath, line_counter)
    

    for filepath in file_list:
        write_unique_lines(filepath, line_counter, output_directory)