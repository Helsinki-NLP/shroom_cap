import json
import csv

# Step 1: Define input and output file paths
input_file = "questions.jsonl"            # original JSONL
updated_jsonl_file = "updated_input.jsonl"  # updated JSONL with answer-reference
csv_file = "output.csv"              # CSV file

updated_rows = []
csv_rows = []

# Step 2: Read JSONL and process entries 
with open(input_file, "r", encoding="utf-8") as infile:
    for line_num, line in enumerate(infile, start=1):
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            print(f"Skipping line {line_num}: invalid JSON format.")
            continue

        title = data.get("title", "").strip()
        question = data.get("question", "").strip()

        print(f"\nEntry {line_num}")
        print(f"Title: {title}")
        print(f"Question: {question}")

        # Ask user for the answer-reference
        answer_ref = input("Enter section number or reference (e.g., 4 or 3.2.2): ").strip()

        # Add the new field without altering other fields
        data["answer-reference"] = f"Section {answer_ref}"

        # Save for JSONL
        updated_rows.append(data)

        # Save for CSV
        csv_rows.append({
            "Title": title,
            "Question": question,
            "Answer Reference": data["answer-reference"]
        })

# Step 3: Write the updated JSONL
with open(updated_jsonl_file, "w", encoding="utf-8") as outjsonl:
    for row in updated_rows:
        json.dump(row, outjsonl, ensure_ascii=False)
        outjsonl.write("\n")
print(f"\nUpdated JSONL file saved as '{updated_jsonl_file}' with {len(updated_rows)} entries.")

# Step 4: Write the CSV
with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Title", "Question", "Answer Reference"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"CSV file '{csv_file}' created with {len(csv_rows)} entries.")
