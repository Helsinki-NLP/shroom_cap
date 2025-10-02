#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict

LANG_MAP = {
    "spanish": "es",
    "english": "en",
    "italian": "it",
    "hindi": "hi",
    "french": "fr",
    "telugu": "te",
    "bengali": "bn",
    "gujarati": "gu",
    "malayalam": "ml"
}

def normalize_label(value, line_num, field_name):
    """
    Checks that value is allowed and converts 'm' / 'minor' -> 'y'.
    """
    if value not in ["y", "n", "m", "minor"]:
        raise ValueError(
            f"Line {line_num}: Invalid label '{value}' in field '{field_name}'. Only 'y', 'n', 'm' or 'minor' are allowed."
        )
    return "y" if value in ["m", "minor"] else value

def balanced_sample(annotations, n=8):
    """
    Balanced sampling across (model_config, prompt, model).
    `annotations` is a list of (line_num, data).
    Returns a list of (line_num, data) up to n=8 items.
    """
    if len(annotations) <= n:
        return annotations

    groups = defaultdict(list)
    for ln, ann in annotations:
        key = (ann.get("model_config"), ann.get("prompt"), ann.get("model"))
        groups[key].append((ln, ann))

    sampled = []
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    while len(sampled) < n and group_keys:
        for key in group_keys[:]:
            if len(sampled) >= n:
                break
            if groups[key]:
                sampled.append(groups[key].pop())
            else:
                group_keys.remove(key)

    # If still not enough (unlikely), fill randomly from remaining
    if len(sampled) < n:
        remaining = []
        for v in groups.values():
            remaining.extend(v)
        if remaining:
            k = min(n - len(sampled), len(remaining))
            sampled.extend(random.sample(remaining, k))

    return sampled

def main():
    parser = argparse.ArgumentParser(
        description="Prepare data: fix indices, clean annotated files, balance annotations, split data/labels."
    )
    parser.add_argument("--language", required=True, help="Language name (e.g., 'spanish').")
    parser.add_argument("--split", required=True, help="Dataset split (e.g., 'train', 'dev', 'test', 'valid').")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for sampling reproducibility.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    language = args.language.lower()
    if language not in LANG_MAP:
        raise ValueError(f"Unsupported language '{language}'. Supported: {list(LANG_MAP.keys())}")

    lang = LANG_MAP[language]
    split_arg = args.split

    # prepare split used in index prefix (do not mutate split_arg)
    if split_arg == 'valid':
        split_for_prefix = 'val'
    elif split_arg == 'test':
        split_for_prefix = 'tst'
    else:
        split_for_prefix = split_arg

    expected_prefix = f"{lang}-{split_for_prefix}-"

    # Input and final output filenames
    input_file = f"{language}/{split_arg}_annotated_data.jsonl"
    cleaned_file = f"{language}/{lang}_{split_arg}_annotated_only.jsonl"
    data_file = f"{language}/{lang}_{split_arg}_data.jsonl"
    label_file = f"{language}/{lang}_{split_arg}_label.jsonl"

    if not os.path.exists(input_file):
        print(f"Error: input file {input_file} not found.")
        return

    # --- Step 0: Load all annotations grouped by paper title, filtering out invalid ones ---
    papers = defaultdict(list)  # title -> list of (line_num, data)
    with open(input_file, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error -> {e}")
                continue

            title = data.get("title")
            if not title:
                # skip lines without a title (or you can choose to log/raise)
                print(f"Warning: Line {line_num} has no 'title' field; skipping.")
                continue

            # Step 0: discard annotations with null label(s)
            fluency = data.get("has_fluency_mistakes")
            factual = data.get("has_factual_mistakes")
            if fluency is None and factual is None:
                # both null => drop
                continue
            elif fluency is None or factual is None:
                # only one null => warn and drop (same behavior as original script)
                print(f"Warning: line {line_num} has only one null field; skipping this annotation.")
                continue

            papers[title].append((line_num, data))

    # --- Step 1: Downsample (balanced) to at most max_annotations per paper ---
    for title in list(papers.keys()):
        papers[title] = balanced_sample(papers[title], n=8)

    indices_data = []
    indices_label = []

    with open(cleaned_file, "w", encoding="utf-8") as out_clean, \
         open(data_file, "w", encoding="utf-8") as out_data, \
         open(label_file, "w", encoding="utf-8") as out_label:

        for title, ann_list in papers.items():
            for line_num, data in ann_list:
                # --- Step 3: Fix index format ---
                if not str(data.get("index", "")).startswith(expected_prefix):
                    # if index missing, this will raise KeyError; keep original behaviour by assuming index exists
                    data["index"] = f"{expected_prefix}{data.get('index', '')}"

                # Remove "lang" field if exists
                data.pop("lang", None)

                # At this point we already filtered out null-fields, so we can normalize safely
                fluency_norm = normalize_label(data.get("has_fluency_mistakes"), line_num, "has_fluency_mistakes")
                factual_norm = normalize_label(data.get("has_factual_mistakes"), line_num, "has_factual_mistakes")

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

    total_written = len(indices_data)
    print(f"Cleaned data written to {cleaned_file}")
    print(f"Data-only file written to {data_file}")
    print(f"Label-only file written to {label_file}")
    print(f"Total annotations written: {total_written}")

    # --- Step 5: Sanity check ---
    if indices_data != indices_label:
        raise ValueError("Index lists of data.jsonl and label.jsonl do not match!")

if __name__ == "__main__":
    main()

