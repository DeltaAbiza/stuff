import json
import os

# --- Configuration ---
# The name of your input JSONL file
input_jsonl_path = 'data/data_clean.jsonl'
# The name of the output text file
output_corpus_path = 'data/corpus.txt'
# The key in the JSON object that contains the text you want to extract
text_key = 'article'


# --- Main Script Logic ---
print(f"Reading from '{input_jsonl_path}' and extracting text...")
try:
    # Open the input and output files
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_corpus_path, 'w', encoding='utf-8') as outfile:

        # Process each line in the JSONL file
        for line in infile:
            try:
                # Load the JSON object from the line
                data = json.loads(line)
                # Extract the text from the specified key
                if text_key in data:
                    article_text = data[text_key]
                    # Write the extracted text to the output file, followed by a newline
                    outfile.write(article_text + '\n')
                else:
                    print(f"Warning: Key '{text_key}' not found in line: {line.strip()}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")

    print(f"\nSuccessfully created corpus file: '{output_corpus_path}'")

except FileNotFoundError:
    print(f"Error: Input file not found at '{input_jsonl_path}'. Please make sure the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

