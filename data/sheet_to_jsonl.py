"""Script to convert spreadsheet entries into a questions.jsonl file.

Requires `pandas` and `openpyxl` for Excel file handling. Typical usage:
python data/sheet_to_jsonl.py \
    --spreadsheet SHROOMCAP-datapoint-creation.xlsx \
    --sheet-name "low-resind" \
    --target-language telugu \
    --reference-jsonl data/papers-with-awards.jsonl \
    --output-folder data/telugu
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_reference_jsonl(path):
    """Load reference JSONL file into a dictionary keyed by title."""
    ref_data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                ref_data[obj["title"]] = obj
    return ref_data


def main():
    """Main function to parse arguments and convert spreadsheet entries to JSONL."""
    parser = argparse.ArgumentParser(description="Convert spreadsheet to questions.jsonl")
    parser.add_argument(
        "--spreadsheet", type=Path, help="Path to the spreadsheet file (Excel or CSV)"
    )
    parser.add_argument("--sheet-name", type=str, help="Target sheet name in the spreadsheet")
    parser.add_argument(
        "--target-language", type=str, help="Target language code (e.g., 'en', 'fr')"
    )
    parser.add_argument("--reference-jsonl", type=Path, help="Path to reference JSONL file")
    parser.add_argument("--output-folder", type=Path, help="Folder to save questions.jsonl")
    args = parser.parse_args()

    ref_data = load_reference_jsonl(args.reference_jsonl)
    df = pd.read_excel(args.spreadsheet, sheet_name=args.sheet_name)

    output_entries = []
    question_col = f"{args.target_language}-question"

    for _, row in df.iterrows():
        title = str(row[0]).strip()
        url = str(row[1]).strip() if not pd.isna(row[1]) else ""
        question = (
            str(row[question_col]).strip()
            if question_col in df.columns and not pd.isna(row[question_col])
            else ""
        )

        if question == "":
            print("Skipping empty question for title:", title)
            continue

        if title in ref_data:
            ref_entry = ref_data[title]
            output_entry = {
                "title": ref_entry.get("title"),
                "abstract": ref_entry.get("abstract"),
                "doi": ref_entry.get("doi"),
                "url": url if url else ref_entry.get("url"),
                "extracted": ref_entry.get("extracted"),
                "datafile": ref_entry.get("datafile"),
                "authors": ref_entry.get("authors"),
                "question": question,
            }
            output_entries.append(output_entry)
        else:
            print(f"Warning: Title '{title}' not found in reference file.")

    output_path = args.output_folder / "questions.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(output_entries)} entries to {output_path}")


if __name__ == "__main__":
    main()
