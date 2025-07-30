import json
from tokenizers import Tokenizer
import os
tokenizer_path = "data/tokenizer.json"
input_data_path = "data/data_clean.jsonl"
output_data_path = "data/tokenize_data.jsonl"
print(f"Loading tokenizer from: {tokenizer_path}")
if not os.path.exists(tokenizer_path):
    print(f"Error: Tokenizer file not found at '{tokenizer_path}'")
    print("Please run the training script first to create it.")
    exit()
tokenizer = Tokenizer.from_file(tokenizer_path)
print("Tokenizer loaded successfully.")
print(f"\nTokenizing data from '{input_data_path}' and saving to '{output_data_path}'...")

if not os.path.exists(input_data_path):
    print(f"Error: Data file not found at '{input_data_path}'")
    exit()

try:
    with open(input_data_path, 'r', encoding='utf-8') as infile, \
         open(output_data_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            data = json.loads(line)
            article = data.get("article", "")
            abstract = data.get("abstract", "")
            article_encoding = tokenizer.encode(article)
            abstract_encoding = tokenizer.encode(abstract)
            tokenized_entry = {
                "article_ids": article_encoding.ids,
                "abstract_ids": abstract_encoding.ids
            }
            outfile.write(json.dumps(tokenized_entry) + '\n')
            if (i + 1) % 100 == 0:
                print(f"  ... processed {i + 1} lines")
            if (i+1) % 6000 == 0:
                break;

    print(f"\nSuccessfully tokenized all data and saved to '{output_data_path}'")

except Exception as e:
    print(f"An error occurred while processing the data file: {e}")
