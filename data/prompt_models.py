#!/usr/bin/env python
# coding: utf-8

import os
ROOT = '..'
os.environ['HF_HOME'] = f'{ROOT}/.hf/'
os.environ['HF_HUB_CACHE'] = f'{ROOT}/.hf/'

import sys
import random
import torch
import tqdm
from transformers.utils import logging
from itertools import product
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


# PREAMBLE:
random.seed(2202)


assert len(sys.argv) > 1, "Missing language input argument.\nCall this script using: \n python3 capture_questions.py <your_language>"
YOUR_LANG = sys.argv[1].lower()
Q_FILE = f"{ROOT}/data/{YOUR_LANG}/questions.jsonl"
OUT_FILE = f'{ROOT}/data/{YOUR_LANG}/generated_answers.jsonl'

MODELS = {
    'spanish': ["Iker/Llama-3-Instruct-Neurona-8b-v2", "meta-llama/Meta-Llama-3-8B-Instruct"],
    'hindi': ["nickmalhotra/ProjectIndus" ,"meta-llama/Meta-Llama-3-8B-Instruct"], #"google/gemma-7b"], #["nickmalhotra/ProjectIndus", "sarvamai/OpenHathi-7B-Hi-v0.1-Base"],
    'french': ["bofenghuang/vigogne-2-13b-chat", "occiglot/occiglot-7b-eu5-instruct", "meta-llama/Meta-Llama-3-8B-Instruct"],
    'italian': ["google/gemma-2-9b-it", "meta-llama/Meta-Llama-3.1-8B-Instruct", "sapienzanlp/modello-italia-9b"],
    # add languages as needed
}
PROMPT_TEMPLATES = {
    'english': {
        'prefix': "In the article titled \"{title}\" by {last},{first} {aux}, ",
        'abstract': "Here is the start of the article abstract for your reference: {abstract}"
    },
    'spanish': {
        'prefix': "En el artículo titulado \"{title}\" escrito por {last},{first} {aux}, ",
        'abstract': "Aquí copio parte del abstract, para que lo uses como referencia: {abstract}"
    },
    'french': {
        'prefix': "Dans l'article intitulé  \"{title}\" ècrit par {first} {last}{aux}, ",
        'abstract': "Voici une partie du résumé de l'article, à titre de référence: {abstract}"
    },
    'hindi': {
        'prefix': "\"{title}\" शीर्षक वाले लेख में {last},{first} {aux} द्वारा, ",
        'abstract': "यहाँ एक संक्षिप्त विवरण है, जहाँ आप संदर्भ के रूप में उपयोग कर सकते हैं: {abstract}"
    },
    'italian': {
        'prefix': "Nell'articolo intitolato \"{title}\" scritto da {first} {last} {aux}, ",
        'abstract': "Ecco la parte iniziale dell'abstract, da usare come riferimento: {abstract}"
    },
    # add languages as needed
}

CONFIGS = [
    ('k50_p0.90_t0.1', dict(top_k=50, top_p=0.90, temperature=0.1)),
    ('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
    ('default', dict()),
]

random.shuffle(CONFIGS)
print('configs used:', CONFIGS)


# copy your hf_token to the ROOT directoy to login fo HF
with open(f'{ROOT}/hf_token', 'r') as file:
    hftoken = file.readlines()[0].strip()

login(token=hftoken, add_to_git_credential=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def build_prompt(row, lang='english', with_abstract=False):
    template = PROMPT_TEMPLATES.get(lang)
    if template is None:
        raise ValueError(f"No prompt template for language: {lang}. Please add a template for it in the PROMPT_TEMPLATES variable")

    aux = ' et al.' if len(row.authors) > 1 else ''
    lowercase_qstart = lambda s: s[:2].lower() + s[2:]

    prompt = template['prefix'].format(
        title=row.title,
        last=row.authors[0]['last'],
        first=row.authors[0]['first'],
        aux=aux
    ) + lowercase_qstart(row.question)

    if with_abstract:
        if not (row.abstract is None):
            prompt += " " + template['abstract'].format(abstract=row.abstract)[:250]

    return prompt


logging.set_verbosity_warning()
records = pd.read_json(Q_FILE, lines=True)
i = 0
n_calls = len(records) * 2 * len(CONFIGS) * len(MODELS[YOUR_LANG])
for model_name in MODELS[YOUR_LANG]:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    for (config_label, config_dict), w_abstract in product(CONFIGS, [False, True]):
        print(f'Prompting: model={model_name}, config={config_label}, prompw_w_abstract={w_abstract}')
        new_records = []
        for _, row in tqdm.tqdm(records.iterrows()):
            prompt = build_prompt(row, lang=YOUR_LANG, with_abstract=w_abstract)
            row = {**row}
            message = [{"role": "user", "content": prompt}]

            inputs = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            attention_mask = torch.ones_like(inputs)  # vector of ones because bsz =1
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                attention_mask=attention_mask,  # transformers complains if this not here
                num_return_sequences=1,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_logits=True,
                do_sample=True,
                **config_dict
            )

            response = outputs.sequences[0][inputs.shape[-1]:]
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            response_token_ids = response.to("cpu").tolist()
            response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
            response_logits = [l.squeeze().to("cpu").tolist()[response_token_ids[idx]] for idx, l in enumerate(outputs.logits)]

            response_text = response_text.replace('\n', '').strip()
            row['model_id'] = model_name
            row['model_config'] = config_label
            row['lang'] = YOUR_LANG
            row['prompt'] = prompt
            row['output_text'] = response_text
            row['output_tokens'] = response_tokens
            row['output_logits'] = response_logits

            with open(OUT_FILE, 'a', encoding='utf-8') as file:
                json.dump(row, file, ensure_ascii=False)
                file.write('\n')

            i += 1
            print(f'prompts done: {i}/{n_calls}')
    model, tokenizer = None, None  # free mem, for next model
    del model, tokenizer
    torch.cuda.empty_cache()
