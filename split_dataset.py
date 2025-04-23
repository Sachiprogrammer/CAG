import json
import gzip
import os
from pathlib import Path
import math

def split_jsonl(input_file, output_dir, chunk_size_mb=100):
    """
    Split a JSONL file into chunks of approximately chunk_size_mb megabytes.
    
    Args:
        input_file (str): Path to input JSONL file (can be gzipped)
        output_dir (str): Directory to save the chunks
        chunk_size_mb (int): Target size of each chunk in megabytes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert MB to bytes
    chunk_size = chunk_size_mb * 1024 * 1024
    
    # Determine if input file is gzipped
    is_gzipped = input_file.endswith('.gz')
    
    # Open input file
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    
    current_chunk = 0
    current_size = 0
    current_file = None
    
    print(f"Processing {input_file}...")
    
    with open_func(input_file, mode) as f:
        for line in f:
            # If we need to start a new chunk
            if current_file is None or current_size >= chunk_size:
                if current_file is not None:
                    current_file.close()
                
                # Create new chunk file (uncompressed)
                chunk_name = f"chunk_{current_chunk:03d}.jsonl"
                chunk_path = os.path.join(output_dir, chunk_name)
                
                current_file = open(chunk_path, 'w', encoding='utf-8')
                current_size = 0
                current_chunk += 1
                print(f"Created new chunk: {chunk_name}")
            
            # Write the line to current chunk
            current_file.write(line)
            current_size += len(line.encode('utf-8'))
    
    # Close the last file
    if current_file is not None:
        current_file.close()
    
    print(f"Split complete. Created {current_chunk} chunks in {output_dir}")

if __name__ == "__main__":
    # Input file path
    input_file = "datasets/googlenaturalquestions/v1.0-simplified_simplified-nq-train.jsonl.gz"
    
    # Output directory
    output_dir = "datasets/googlenaturalquestions/chunks"
    
    # Split the dataset
    split_jsonl(input_file, output_dir, chunk_size_mb=100) 