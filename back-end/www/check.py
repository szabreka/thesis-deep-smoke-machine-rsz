import numpy as np
import os

# Directory containing the .npy files
data_dir = "/../../../projects/0/prjs0930/data/rgb/"

# List all files in the directory
files = os.listdir(data_dir)

# Process only .npy files
npy_files = [f for f in files if f.endswith('.npy')]

expected_shape = (36, 180, 180, 3)

# Iterate over each .npy file
for file in npy_files:
    file_path = os.path.join(data_dir, file)
    try:
        data = np.load(file_path)
        if data.shape != expected_shape:
            print(f"File: {file}")
            print(f"Shape: {data.shape}")
            print(f"Size: {data.size}")
    except Exception as e:
        print(f"Error loading file {file}: {e}")
