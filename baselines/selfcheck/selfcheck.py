from datasets import load_dataset
import json
import os
import random
import numpy as np
import tqdm
seed_val = 442
random.seed(seed_val)
np.random.seed(seed_val)

from huggingface_hub import login

import torch
from transformers import pipeline

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

HUGGINGFACE_TOKEN = "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"
login(token=HUGGINGFACE_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pipe = pipeline("text-generation", model="google/gemma-2-9b", device=device)

# Process each language
for lang in ["en", "it", "es", "fr", "hi"]:
    path_val_file = f"../../data/release_folder/validation/{lang}_valid_data.jsonl"
    FILENAME = os.path.basename(path_val_file)
    path_val_output = f"{lang}_valid_label_selfcheck.jsonl"

    print(f"Processing language: {lang}")

    data_val_all = []
    with open(path_val_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data_val_all.append(json.loads(line))

    num_sample = len(data_val_all)
    print(num_sample)

    output_json = []
    labels = ["n", "y"]

    for ii in tqdm.tqdm(range(num_sample)):
        row = data_val_all[ii]
        curr_output = {'index': row['index'], 'has_fluency_mistakes': '', 'has_factual_mistakes': ''}

        tasks = ["hallucination", "fluency mistakes"]

        for ki, kv in enumerate(['has_factual_mistakes', 'has_fluency_mistakes']):

            prefix = f"""
            As a Self-CheckLLM judge, read the passage (title, abstract, question, answer) and decide if the answer contains {tasks[ki]} or not , ONLY OUTPUT YES OR NO:
            """

            passage = f"""
            title : [{row['title']}]

            abstract : [{row['abstract']}]

            question : [{row['question']}]

            answer : [{row['output_text']}]

            Answer : 
            """

            prompt = prefix + passage    

            RESPONSE = pipe(prompt, do_sample=False, max_new_tokens=128, return_full_text=False)
            answer =  RESPONSE[0]['generated_text'].lower()

            if answer.startswith("yes") or answer.__contains__("yes"):
                output_label = "y"
                # prob = 1-float(np.exp(response["choices"][0]["logprobs"]["token_logprobs"][0]))
            elif answer.startswith("no") or answer.__contains__("no"):
                output_label = "n"
               # prob = float(np.exp(response["choices"][0]["logprobs"]["token_logprobs"][0]))
            elif (not answer.startswith("no") and not answer.startswith("yes")) or (not answer.__contains__("yes") and not answer.__contains__("no")):
                idx_random = random.randint(0,len(labels)-1)
                output_label = labels[idx_random]
                # prob = float(0.5)

            curr_output[kv] = output_label

        output_json.append(curr_output)


with open(path_val_output, "w", encoding="utf-8") as f:
    for record in output_json:  # output_json is a list of dicts
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

