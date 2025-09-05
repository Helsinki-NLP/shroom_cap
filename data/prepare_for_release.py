import argparse
import json
import os

LANG_MAP = {
    "spanish": "es",
    "english": "en",
    "italian": "it",
    "hindi": "hi",
    "french": "fr"
}

def normalize_label(value, line_num, field_name):
    """
    Checks that value is allowed and converts 'm' -> 'y'.
    """
    if value not in ["y", "n", "m"]:
        raise ValueError(f"Line {line_num}: Invalid label '{value}' in field '{field_name}'. Only 'y', 'n', 'm' are allowed.")
    return "y" if value == "m" else value
    
def main():
    parser = argparse.ArgumentParser(description="Prepares data for relese by fixing indices, cleaning annotated files and splitting data/labels.")
    parser.add_argument("--language", required=True, help="Language name (e.g., 'spanish').")
    parser.add_argument("--split", required=True, help="Dataset split (e.g., 'train', 'dev', 'test').")
    args = parser.parse_args()

    language = args.language.lower()
    if language not in LANG_MAP:
        raise ValueError(f"Unsupported language '{language}'. Supported: {list(LANG_MAP.keys())}")

    lang = LANG_MAP[language]
    split = args.split

    # Input and final output filenames
    input_file = f"{language}/{split}_annotated_data.jsonl"
    cleaned_file = f"{language}/{lang}_{split}_annotated_only.jsonl"
    data_file = f"{language}/{lang}_{split}_data.jsonl"
    label_file = f"{language}/{lang}_{split}_label.jsonl"

    if not os.path.exists(input_file):
        print(f"Error: input file {input_file} not found.")
        return
        
    indices_data = []
    indices_label = []

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(cleaned_file, "w", encoding="utf-8") as out_clean, \
         open(data_file, "w", encoding="utf-8") as out_data, \
         open(label_file, "w", encoding="utf-8") as out_label:

        for line_num, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error -> {e}")
                continue

            # --- Step 1: Fix index format ---
            expected_prefix = f"{lang}-{split}-"
            if not str(data.get("index", "")).startswith(expected_prefix):
                data["index"] = f"{expected_prefix}{data['index']}"

            # Remove "lang" field if exists
            data.pop("lang", None)

            # --- Step 2: Discard null fields ---
            fluency = data.get("has_fluency_mistakes")
            factual = data.get("has_factual_mistakes")

            if fluency is None and factual is None:
                continue
            elif fluency is None or factual is None:
                print(f"Warning: line {line_num} has only one null field.")
                continue
                
            # --- Step 3: Validate and normalize labels ---
            fluency_norm = normalize_label(fluency, line_num, "has_fluency_mistakes")
            factual_norm = normalize_label(factual, line_num, "has_factual_mistakes")

            # --- Step 4: Write outputs ---
            # Full cleaned file (annotations only, correct indexing)
            out_clean.write(json.dumps(data, ensure_ascii=False) + "\n")

            # Data-only file
            data_copy = {k: v for k, v in data.items() if k not in ["has_fluency_mistakes", "has_factual_mistakes"]}
            out_data.write(json.dumps(data_copy, ensure_ascii=False) + "\n")
            indices_data.append(data_copy["index"])

            # Labels-only file
            label_copy = {
                "index": data["index"],
                "has_fluency_mistakes": fluency_norm,
                "has_factual_mistakes": factual_norm
            }
            out_label.write(json.dumps(label_copy, ensure_ascii=False) + "\n")
            indices_label.append(label_copy["index"])

    print(f"Cleaned data written to {cleaned_file}")
    print(f"Data-only file written to {data_file}")
    print(f"Label-only file written to {label_file}")
    
    # --- Step 5: Sanity check ---
    if indices_data != indices_label:
        raise ValueError("Undex lists of data.jsonl and label.jsonl do not match!")

if __name__ == "__main__":
    main()

