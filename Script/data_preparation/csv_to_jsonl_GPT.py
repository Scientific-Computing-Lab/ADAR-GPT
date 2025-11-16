#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV → JSONL (minimal changes version)
------------------------------------
- Input path provided via CLI (no hard-coded paths).
- Optional output path via CLI; if not provided, the original name-derivation logic is used.
- Comments/prints in English.

Behavior intentionally preserved:
- Assumes CSV has a header row.
- Expects four columns in each data row: structure, L, R, y_n (label).
- Builds user content including the structure (as in the original).
"""

import csv
import json
import os
import argparse

def csv_to_jsonl(input_filename, output_jsonl=None):
    # Derive base name and directory 
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    directory = os.path.dirname(input_filename)

    # Extract the last part after the last underscore 
    base_name_parts = base_name.split("_")
    print(base_name_parts)
    short_name = base_name_parts[0] if len(base_name_parts) > 1 else base_name

    # If output not provided, reproduce original filename pattern:
    # f"{short_name}_{base_name_parts[1]}.jsonl" when parts[1] exists;
    # otherwise fall back to f"{short_name}.jsonl"
    if output_jsonl is None:
        if len(base_name_parts) > 1:
            training_filename = os.path.join(directory, f"{short_name}_{base_name_parts[1]}.jsonl")
        else:
            training_filename = os.path.join(directory, f"{short_name}.jsonl")
    else:
        training_filename = output_jsonl

    system_message = {
        "role": "system",
        "content": "Predict if the central adenosine (A) in the given RNA sequence context within an Alu element will be edited to inosine (I) by ADAR enzymes."
    }

    data = []

    # Read CSV exactly like the original (header + rows)
    with open(input_filename, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Skip the header row if there is one

        for row in csv_reader:
            # Expecting 4 columns: structure, L, R, y_n
            structure, L, R, y_n = row
            contant = f"L:{L}, A:A, R:{R}, Alu Vienna Structure:{structure}"
            data.append({
                "messages": [
                    system_message,
                    {"role": "user", "content": contant},
                    {"role": "assistant", "content": y_n}
                ]
            })

    # Write JSONL (same as original)
    with open(training_filename, mode='w') as train_file:
        for entry in data:
            train_file.write(json.dumps(entry) + '\n')

    print(f"Training data saved to {training_filename}.")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to JSONL for GPT training (minimal-change version).")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--output_jsonl", default=None, help="Optional output JSONL path. If omitted, original naming logic is used.")
    args = parser.parse_args()

    csv_to_jsonl(input_filename=args.input_csv, output_jsonl=args.output_jsonl)

if __name__ == "__main__":
    main()
