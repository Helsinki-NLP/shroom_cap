import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

model_name = "fava-uw/fava-model"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="/ssd_scratch/patanjali.b/models"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="/ssd_scratch/patanjali.b/models",
    torch_dtype="auto",
    device_map="auto"   
)
model.eval()

# ---------------- Load dataset ----------------
data = pd.read_pickle("multilingual_data_with_labels.pkl")
print("Columns:", data.columns)

# ---------- Chunking ----------
def chunk_text(text, chunk_size=350, overlap=20):
    """
    Split text into overlapping chunks of tokens (split by whitespace).
    """
    tokens = text.split()
    for i in range(0, len(tokens), chunk_size - overlap):
        yield " ".join(tokens[i:i+chunk_size])
        
# ---------- Inference ----------

def detect_hallucination(fava_output: str) -> str:
    """Return 'y' if hallucination (<delete> tags) present, else 'n'."""
    if re.search(r"<delete>.*?</delete>", fava_output, flags=re.DOTALL):
        return "y"
    return "n"

def run_fava_inference(df, model, tokenizer, batch_size=1, device="cuda"):
    """
    Run hallucination detection with FAVA model using chunking + batching.
    Memory-optimized version.
    """
    all_outputs, hallucination_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pdf_text = row["pdf_text"]
        hypothesis = row["output_text"]

        prompts = []
        for chunk in chunk_text(pdf_text, chunk_size=128, overlap=20):  # smaller chunks
            prompts.append(
                f"Read the following references:\n{chunk}\n"
                f"Please identify all the errors in the following text "
                f"using the information in the references provided and suggest edits if necessary:\n"
                f"[Text] {hypothesis}\n[Edited] "
            )

        chunk_outputs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,   # reduce from 512
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    use_cache=False       # saves VRAM
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            cleaned = [d.split("[Edited]")[-1].strip() for d in decoded]
            chunk_outputs.extend(cleaned)

            del inputs, outputs  # free memory
            torch.cuda.empty_cache()

        halluc_flags = [detect_hallucination(out) for out in chunk_outputs]
        final_flag = "y" if "y" in halluc_flags else "n"

        all_outputs.append(" ||| ".join(chunk_outputs))
        hallucination_labels.append(final_flag)

    df["fava_output"] = all_outputs
    df["hallucinated"] = hallucination_labels
    return df

if __name__ == "__main__":
    results = run_fava_inference(data, model, tokenizer, batch_size=2, device="cuda")
    results.to_json("fava_hallucination_results.json", orient="records", lines=True)
    
