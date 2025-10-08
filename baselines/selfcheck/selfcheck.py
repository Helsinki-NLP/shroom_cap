import json
import random
import numpy as np
import tqdm
import argparse
seed_val = 442
random.seed(seed_val)
np.random.seed(seed_val)

from huggingface_hub import login

from transformers import pipeline

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64


def main():
    parser = argparse.ArgumentParser(description="Run self-check baseline for a specific language")
    parser.add_argument(
        "--lang",
        required=True,
        choices=["en", "it", "es", "fr", "hi", "bn", "gu", "ml", "te"],
        help="Language code."
    )
    parser.add_argument("--hf_token", required=True, help="HuggingFace token")
    args = parser.parse_args()

    # Set seeds
    seed_val = 442
    random.seed(seed_val)
    np.random.seed(seed_val)

    login(token=args.hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    pipe = pipeline("text-generation", model="google/gemma-2-9b", device=device)

    lang = args.lang
    path_file = f"../../data/release_folder/test/{lang}_test_data.jsonl"
    path_output = f"{lang}_test_label_selfcheck.jsonl"

    print(f"[INFO] Processing language: {lang}")

    # Load data
    data_all = []
    with open(path_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data_all.append(json.loads(line))

    num_sample = len(data_all)
    print(f"[INFO] Number of samples: {num_sample}")

    output_json = []
    labels = ["n", "y"]
    tasks = ["hallucination", "fluency mistakes"]
    keys = ['has_factual_mistakes', 'has_fluency_mistakes']

    for row in tqdm.tqdm(data_all):
        curr_output = {'index': row['index'], 'has_fluency_mistakes': '', 'has_factual_mistakes': ''}

        for task, key in zip(tasks, keys):
            prompt = f"""
            As a Self-CheckLLM judge, read the passage (title, abstract, question, answer) 
            and decide if the answer contains {task} or not, ONLY OUTPUT YES OR NO:

            title : [{row['title']}]

            abstract : [{row['abstract']}]

            question : [{row['question']}]

            answer : [{row['output_text']}]

            Answer :
            """
            response = pipe(prompt, do_sample=False, max_new_tokens=128, return_full_text=False)
            answer = response[0]['generated_text'].lower().strip()

            if "yes" in answer:
                output_label = "y"
            elif "no" in answer:
                output_label = "n"
            else:
                output_label = random.choice(labels)

            curr_output[key] = output_label

        output_json.append(curr_output)

    # Save output
    with open(path_output, "w", encoding="utf-8") as f:
        for record in output_json:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Results saved to {path_output}")


if __name__ == "__main__":
    main()


