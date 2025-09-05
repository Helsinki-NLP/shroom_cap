import json
import random
import argparse
import time


def create_random_baseline(input_path, output_path, seed=None):
    """
    Create a random baseline .jsonl file from an existing dataset.
    
    Args:
        input_path (str): Path to the input .jsonl file
        output_path (str): Path to the output .jsonl file with random predictions
        seed (int, optional): Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    else:
        seed = int(time.time())
        random.seed(seed)
        print(f"[INFO] No seed provided. Using auto-generated seed: {seed}")

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            entry = json.loads(line)
            random_entry = {
                "index": entry["index"],
                "has_fluency_mistakes": random.choice(["y", "n"]),
                "has_factual_mistakes": random.choice(["y", "n"])
            }
            outfile.write(json.dumps(random_entry) + "\n")

    print(f"[INFO] Random baseline saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random baseline JSONL prediction file."
    )
    parser.add_argument("--input", help="Path to input JSONL file")
    parser.add_argument("--output", help="Path to save random baseline JSONL file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: auto)")
    
    args = parser.parse_args()
    create_random_baseline(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
