import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from hdm2 import HallucinationDetectionModel
import contextlib
import io
# ---------------- Load dataset ----------------
data = pd.read_pickle("multilingual_data_with_labels.pkl")
print("Columns:", data.columns)
# Expecting columns: ["question", "pdf_text", "output_text", "language", "label"]

def chunk_text(text, chunk_size=350, overlap=20):
    """
    Split text into overlapping chunks of tokens (split by whitespace).
    """
    tokens = text.split()
    for i in range(0, len(tokens), chunk_size - overlap):
        yield " ".join(tokens[i:i+chunk_size])

# ---------------- Tokenizer ----------------
hdm_model = HallucinationDetectionModel()
print("Detecting hallucinations with default parameters...")

def run_inference(df, detector, batch_size=8, threshold=0.5):
    """
    df: dataframe with columns ['question', 'pdf_text', 'output_text']
    detector: HDM2 HallucinationDetectionModel
    tokenizer: the same tokenizer used in the model
    batch_size: batch size for inference
    threshold: hallucination score threshold to mark 'y'/'n'
    """
    all_scores, hallucination_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question, pdf_text, hypothesis = row["question"], row["pdf_text"], row["output_text"]

        # Prepare premise-hypothesis prompts with chunking
        prompts = []
        for chunk in chunk_text(pdf_text, chunk_size=800):
            prompt_text = f"{question} {chunk}"
            prompts.append((prompt_text, hypothesis))

        # Run inference in batches
        chunk_scores = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]

            for context_text, response_text in batch:
                # Use your HDM2 detector
                with contextlib.redirect_stdout(io.StringIO()):
                    results = detector.apply(
                        "You are an AIMon Bot. Help me classify hallucinations.",
                        context_text,
                        response_text
                    )

                # Adjust this depending on HDM2 output structure
                chunk_score = results.get("adjusted_hallucination_severity", 0.0)
                chunk_scores.append(chunk_score)

                torch.cuda.empty_cache()
                del results, context_text, response_text
        

        # Take max score across chunks as row-level hallucination score
        row_score = max(chunk_scores) if chunk_scores else 0.0
        all_scores.append(row_score)
        hallucination_labels.append("y" if row_score >= threshold else "n")

    df["hallucination_score"] = all_scores
    df["hallucinated"] = hallucination_labels
    return df

# Assume `data` is your dataframe
results = run_inference(data, hdm_model, batch_size=2, threshold=0.5)

# Save results
results.to_json("hdm2_results.json", orient="records", lines=True)

