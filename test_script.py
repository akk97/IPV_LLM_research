# ~/test_script.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === CONFIG ===
model_cache = "/scratch/pioneer/jobs/rxs1540/hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
excel_file = "/home/akk97/Quotations.xlsx"
column_name = "quotations_cleaned"  # change if different in your Excel

# === LOAD MODEL & TOKENIZER ===
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_cache, use_fast=True, trust_remote_code=True)

# Ensure pad_token exists for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_cache)

# === SET UP PIPELINE ===
age_predictor = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,          # set to -1 if using CPU
    trust_remote_code=True,
)

# === READ EXCEL ===
print("Reading Excel file...")
df = pd.read_excel(excel_file)
if column_name not in df.columns:
    raise ValueError(f"Column '{column_name}' not found. Available: {df.columns.tolist()}")

# === HELPER FUNCTION ===
def predict_age_batch(texts, max_new_tokens=16):
    """Return predicted ages for a batch of texts"""
    prompts = [f"Guess the age of the person speaking: {q}" for q in texts]
    outputs = age_predictor(prompts, max_new_tokens=max_new_tokens, batch_size=len(prompts))
    # Extract first line of output
    ages = [o[0]["generated_text"].split(":")[-1].strip() for o in outputs]
    return ages

# === RUN BATCH PREDICTIONS ===
print("Predicting ages in batches...")
batch_size = 8
predicted_ages = []

for i in range(0, len(df), batch_size):
    batch = df[column_name].iloc[i:i+batch_size].astype(str).tolist()
    predicted_ages.extend(predict_age_batch(batch))

# === SAVE RESULTS ===
df["Predicted_Age"] = predicted_ages
output_file = excel_file.replace(".xlsx", "_with_ages.xlsx")
df.to_excel(output_file, index=False)
print(f"Done! Saved predictions to {output_file}")
