import json
import argparse

parser = argparse.ArgumentParser(description="Modify JSONL file index format")
parser.add_argument("--lang", required=True, help="Language code, e.g., 'en'")
args = parser.parse_args()
lang = args.lang

# Input and output files
input_file = f"{lang}_dev_annotated_data.jsonl"
output_file = f"{lang}_dev_annotated_data_updated.jsonl"

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if line.strip():  # skip empty lines
            data = json.loads(line)
            
            # Modify the index field
            data['index'] = f"{lang}-val-{data['index']}"
            
            # Remove lang field
            if 'lang' in data:
                del data['lang']
            
            # Write the modified line
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"Modified inxed in {input_file} --> {output_file}")