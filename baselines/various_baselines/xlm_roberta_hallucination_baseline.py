import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


data = pd.read_pickle("multilingual_test_data_with_labels.pkl")
data.head()

print(data.columns)

import torch
from transformers import pipeline

hallucination_detector = pipeline(
    task='text-classification',
    model='bond005/xlm-roberta-xl-hallucination-detector',
    framework='pt', trust_remote_code=True, device='cuda', torch_dtype=torch.float16
)

def chunk_text(text, chunk_size=350, overlap=20):
    """
    Split text into overlapping chunks of tokens (split by whitespace).
    """
    tokens = text.split()
    for i in range(0, len(tokens), chunk_size - overlap):
        yield " ".join(tokens[i:i+chunk_size])

# --- Inference ---
def run_inference(df, detector, batch_size=8, threshold=0.5):
    """
    Run hallucination detection row by row with chunking + batching.

    Args:
        df: DataFrame with ['question', 'pdf_text', 'output_text']
        detector: HuggingFace pipeline (hallucination detector)
        batch_size: number of pairs per forward pass
        threshold: classification threshold (default 0.5)

    Returns:
        DataFrame with new columns ['hallucination_score', 'hallucinated']
    """
    all_scores, hallucination_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        pdf_text = row["pdf_text"]
        hypothesis = row["output_text"]

        # Build premise-hypothesis pairs
        pairs = []
        for chunk in chunk_text(pdf_text, chunk_size=150):
            premise = question + " " + chunk
            pairs.append(f"Premise: {premise}\nHypothesis: {hypothesis}")

        # Batched predictions
        chunk_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            predictions = detector(batch, truncation=True, max_length=514)

            for pred in predictions:
                if pred["label"] == "Hallucination":
                    score = pred["score"]
                else:
                    score = 1.0 - pred["score"]
                chunk_scores.append(score)

            torch.cuda.empty_cache()

        # Aggregate: take maximum across chunks
        row_score = max(chunk_scores) if chunk_scores else 0.0
        all_scores.append(row_score)

        # Apply threshold
        hallucination_labels.append("y" if row_score >= threshold else "n")

    df["hallucination_score"] = all_scores
    df["hallucinated"] = hallucination_labels
    return df

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd

def compute_metrics(df, label_col="label", pred_col="hallucinated"):
    """
    Compute precision, recall, f1, accuracy overall + per language.
    
    CRITICAL FILTER APPLIED: Only computes metrics where:
    1. df['fluency_mistake'] == 'n' (Ground Truth is Fluent)
    """
    
    # Check for required column before filtering
    if 'fluency_mistake' not in df.columns:
        print("Error: 'fluency_mistake' column required for metric filtering but not found.")
        return pd.DataFrame() # Return empty DataFrame if column is missing

    # --- APPLY FILTER ---
    # FIX 2: Filter the DataFrame to include only fluent responses.
    df_filtered = df[df['fluency_mistake'].astype(str) == 'n'].copy()
    
    if df_filtered.empty:
        print("Warning: Filtered DataFrame is empty after filtering by fluency='n'. Cannot compute metrics.")
        return pd.DataFrame()
        
    results = {}
    
    # Ensure language column exists for grouping
    if "language" not in df_filtered.columns and "index" in df_filtered.columns:
        df_filtered["language"] = df_filtered["index"].astype(str).str[:2]
    
    # Map 'y' to 1 and 'n' to 0 for sklearn metrics
    y_true_all = df_filtered[label_col].map({"y": 1, "n": 0}).values
    y_pred_all = df_filtered[pred_col].map({"y": 1, "n": 0}).values

    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="binary", zero_division=0)
    acc = accuracy_score(y_true_all, y_pred_all)
    results["overall"] = {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc, "support": len(df_filtered)}

    # Per-language metrics
    if "language" in df_filtered.columns:
        for lang, subdf in df_filtered.groupby("language"):
            if len(subdf) == 0:
                continue

            y_true_lang = subdf[label_col].map({"y": 1, "n": 0}).values
            y_pred_lang = subdf[pred_col].map({"y": 1, "n": 0}).values
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_true_lang, y_pred_lang, average="binary", zero_division=0)
            acc = accuracy_score(y_true_lang, y_pred_lang)
            results[lang] = {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc, "support": len(subdf)}

    # Return as DataFrame
    return pd.DataFrame(results).T


df = run_inference(data, hallucination_detector, batch_size=8, threshold=0.5)

df.to_json("xlm_roberta_hallucination_results.json", orient="records", lines=True)

results  = compute_metrics(df, label_col="factual_mistake", pred_col="hallucinated")

results.to_csv("xlm_roberta_hallucination_metrics.csv")

