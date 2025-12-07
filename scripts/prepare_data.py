"""
Data Preparation Script

Cleans and prepares parallel corpus data for training.

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import argparse
import re


def normalize_sentence(sentence):
    """
    Normalize a sentence: lowercase, trim, remove extra spaces
    
    Author: Molla Samser (https://rskworld.in)
    """
    sentence = sentence.lower().strip()
    sentence = re.sub(r'\s+', ' ', sentence)  # Remove multiple spaces
    return sentence


def clean_data(input_file, output_file, min_length=3, max_length=100):
    """
    Clean and prepare parallel corpus data
    
    Author: Molla Samser (https://rskworld.in)
    """
    cleaned_pairs = []
    skipped = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip comment lines
            if line.strip().startswith('#'):
                continue
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Split source and target
            if '|||' in line:
                parts = line.strip().split('|||')
                if len(parts) == 2:
                    src = normalize_sentence(parts[0])
                    tgt = normalize_sentence(parts[1])
                    
                    # Filter by length
                    src_words = src.split()
                    tgt_words = tgt.split()
                    
                    if (min_length <= len(src_words) <= max_length and
                        min_length <= len(tgt_words) <= max_length):
                        cleaned_pairs.append(f"{src} ||| {tgt}\n")
                    else:
                        skipped += 1
                else:
                    skipped += 1
            else:
                skipped += 1
    
    # Write cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_pairs)
    
    print(f"Author: Molla Samser - https://rskworld.in")
    print(f"Cleaned {len(cleaned_pairs)} sentence pairs")
    print(f"Skipped {skipped} pairs (length or format issues)")
    print(f"Output saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare and clean parallel corpus data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input data file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output cleaned data file')
    parser.add_argument('--min_length', type=int, default=3,
                       help='Minimum sentence length (words)')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum sentence length (words)')
    
    args = parser.parse_args()
    
    clean_data(args.input, args.output, args.min_length, args.max_length)

