import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
import pandas as pd

data = pd.read_pickle("multilingual_data_with_labels.pkl")
print(data.columns)

# ---------- Chunking ----------
def chunk_text(text, chunk_size=350, overlap=20):
    """
    Split text into overlapping chunks of tokens (split by whitespace).
    """
    tokens = text.split()
    for i in range(0, len(tokens), chunk_size - overlap):
        yield " ".join(tokens[i:i+chunk_size])
        
# --------- Load generation model (LLM) ---------
gen_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_name, 
    device_map="auto", 
    torch_dtype=torch.float16,
    use_auth_token="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",
    cache_dir="/ssd_scratch/patanjali.b/models",
)

# --------- Load NLI model (multilingual) ---------
nli_model_name = "joeddav/xlm-roberta-large-xnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    nli_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir="/ssd_scratch/patanjali.b/models"
)

id2label = nli_model.config.id2label  # {0: contradiction, 1: neutral, 2: entailment}
print(id2label)


# --------- Generation function ---------
def generate_samples(prompt, k=5, max_new_tokens=256):
    """Generate k stochastic samples for a given prompt."""
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    outputs = gen_model.generate(
        **inputs,
        do_sample=True,
        top_p=0.95,
        temperature=0.2,
        max_new_tokens=max_new_tokens,
        num_return_sequences=k
    )
    return [gen_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

# --------- NLI-based SelfCheck ---------
def selfcheck_nli(original_answer, re_samples):
    consistent = 0
    for sample in re_samples:
        encoding = nli_tokenizer(
            sample,
            original_answer,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(nli_model.device)

        with torch.no_grad():
            logits = nli_model(**encoding).logits
            pred = torch.argmax(logits, dim=1).item()
            label = id2label[pred]

        if label.lower() == "entailment":
            consistent += 1

    return consistent / len(re_samples) if re_samples else 0.0

# --------- Runner for DataFrame ---------
def run_selfcheck(df, k=5):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"]
        original_answer = row["output_text"]

        re_samples = generate_samples(prompt, k=k)
        score = selfcheck_nli(original_answer, re_samples)

        results.append({
            "index": row["index"],
            "prompt": prompt,
            "original_answer": original_answer,
            "selfcheck_score": score,
            "predicted_label": "y" if score < 0.5 else "n",
            "fluency_mistake": row.get("fluency_mistake", None),
            "factual_mistake": row.get("factual_mistake", None)
        })
    return pd.DataFrame(results)

# --------- Execute SelfCheck on the dataset ---------

# --------- Example usage ---------
# Suppose your dataframe is already loaded as `data`
selfcheck_results = run_selfcheck(data, k=5)

# Save to file
selfcheck_results.to_json("selfcheck_results.jsonl", orient="records", lines=True)

# Also inspect first few rows
print(selfcheck_results.head())


