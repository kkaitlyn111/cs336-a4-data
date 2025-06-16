import numpy as np

# Paths to the input files
file1 = "/home/user/cs336-a4-data/tokenized_outputs/example.npy"
file2 = "/home/user/cs336-a4-data/tokenized_outputs/00892.npy"

# Path to the output file
output_file = "/home/user/cs336-a4-data/tokenized_outputs/combined.npy"

# Load the arrays
arr1 = np.fromfile(file1, dtype=np.uint16)
arr2 = np.fromfile(file2, dtype=np.uint16)

# Concatenate the arrays
combined = np.concatenate([arr1, arr2])

# Save the combined array
combined.tofile(output_file)

print(f"Combined {file1} and {file2} into {output_file} with {len(combined)} tokens.") 