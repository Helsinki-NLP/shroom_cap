import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ---------------- Load dataset ----------------
data = pd.read_pickle("multilingual_data_with_labels.pkl")
print("Columns:", data.columns)
# Expecting columns: ["question", "pdf_text", "output_text", "language", "label"]

# ---------------- Tokenizer ----------------
model_name = "bond005/xlm-roberta-xl-hallucination-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def chunk_text(text, tokenizer, chunk_size=400, overlap=50):
    """
    Tokenizer-based chunking to respect model's max_length (514).
    """
    tokens = tokenizer.tokenize(text)
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i+chunk_size]
        yield tokenizer.convert_tokens_to_string(chunk)

# ---------------- Pipeline ----------------
hallucination_detector = pipeline(
    task="text-classification",
    model=model_name,
    tokenizer=tokenizer,
    device_map=0,  # Use all available GPUs
    torch_dtype=torch.float16,
    trust_remote_code=True,
    truncation=True,
    max_length=512
)

# ---------------- Inference ----------------
def run_inference(df, detector, tokenizer, batch_size=8, threshold=0.5):
    all_scores, hallucination_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question, pdf_text, hypothesis = row["question"], row["pdf_text"], row["output_text"]

        # Premise-hypothesis pairs (tokenizer-safe chunking)
        pairs = []
        for chunk in chunk_text(pdf_text, tokenizer, chunk_size=200):
            premise = question + " " + chunk
            pairs.append(f"Premise: {premise}\nHypothesis: {hypothesis}")

        chunk_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            predictions = detector(batch)

            for pred in predictions:
                score = pred["score"] if pred["label"] == "Hallucination" else (1.0 - pred["score"])
                chunk_scores.append(score)

            torch.cuda.empty_cache()

        row_score = max(chunk_scores) if chunk_scores else 0.0
        all_scores.append(row_score)
        hallucination_labels.append("y" if row_score >= threshold else "n")

    df["hallucination_score"] = all_scores
    df["hallucinated"] = hallucination_labels
    return df

# ---------------- Metrics ----------------
def compute_metrics(df, label_col="label", pred_col="hallucinated"):
    """
    Compute precision, recall, f1, accuracy overall + per language
    """
    results = {}
    y_true = df[label_col].map({"y": 1, "n": 0}).values
    y_pred = df[pred_col].map({"y": 1, "n": 0}).values

    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    acc = accuracy_score(y_true, y_pred)
    results["overall"] = {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

    # Per-language metrics
    if "language" in df.columns:
        for lang, subdf in df.groupby("language"):
            y_true = subdf[label_col].map({"y": 1, "n": 0}).values
            y_pred = subdf[pred_col].map({"y": 1, "n": 0}).values
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
            acc = accuracy_score(y_true, y_pred)
            results[lang] = {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

    return results

# ---------------- Run ----------------
if __name__ == "__main__":
    results = run_inference(data, hallucination_detector, tokenizer, batch_size=4, threshold=0.5)
    results.to_json("xlm_roberta_hallucination_results.json", orient="records", lines=True)

    metrics = compute_metrics(results)
    pd.DataFrame(metrics).T.to_csv("xlm_roberta_hallucination_metrics.csv")
    print("Saved predictions and metrics.")
    print(metrics)
