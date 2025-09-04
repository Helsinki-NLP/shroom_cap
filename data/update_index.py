import json
import argparse

parser = argparse.ArgumentParser(description="Modify JSONL file index format")
parser.add_argument("--language", required=True, help="Language code, e.g., 'en'")
parser.add_argument("--split", required=True, help="Dataset split, e.g., 'train', 'dev', 'test', 'train2'")

args = parser.parse_args()

language = args.language
split = args.split

language = language.lower()

# Input and output files
input_file = f"{language}/{split}_annotated_data.jsonl"
output_file = f"{language}/{language[:2]}_{split}_annotated_data.jsonl"

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if line.strip():  # skip empty lines
            data = json.loads(line)
            
            # Modify the index field
            data['index'] = f"{language[:2]}-{split}-{data['index']}"
            
            # Remove lang field
            if 'lang' in data:
                del data['lang']
            
            # Write the modified line
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"Modified inxed in {input_file} --> {output_file}")

