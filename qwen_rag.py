import pandas as pd
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
 
# =========================
# CONFIG
# =========================
model_cache = "/scratch/pioneer/jobs/akk97/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
lora_path = "/home/akk97/lora_4tasks_binaryrelevance_adapter_v3_qwen/checkpoint-1041"
 
excel_file = "/home/akk97/IPVData_cleaned.xlsx"
column_name = "quotations_cleaned"
title_col = "Title of the articles"
abstract_col = "abstracts"
 
output_progress = "rag_qwen_labels_progress.xlsx"
 
IPV_INDICATORS = [
    "Housing_Shelter",
    "Mental_Health",
    "Professional_Help_Seeking",
    "Location",
    "Neighborhood_Safety_Crime",
    "Violence",
    "Law",
    "Family",
    "Income",
    "Reproductive_Care",
]
 
# =========================
# LOAD MODEL (WITH LORA)
# =========================
print("Loading model and tokenizer...")
 
tokenizer = AutoTokenizer.from_pretrained(
    model_cache,
    use_fast=True,
    trust_remote_code=True
)
 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
 
base_model = AutoModelForCausalLM.from_pretrained(
    model_cache,
    device_map="auto",
    torch_dtype="auto"
)
 
model = PeftModel.from_pretrained(
    base_model,
    lora_path
)
model.eval()
 
print("Model loaded successfully.")
 
# =========================
# GET TRUE/FALSE TOKEN IDS
# =========================
def get_bool_token_ids(tokenizer):
    candidates = {
        "true":  [" True", "True",  " true",  "true"],
        "false": [" False", "False", " false", "false"],
    }
    ids = {}
    for label, variants in candidates.items():
        for v in variants:
            toks = tokenizer.encode(v, add_special_tokens=False)
            if len(toks) == 1:
                ids[label] = toks[0]
                break
        if label not in ids:
            ids[label] = tokenizer.encode(variants[0], add_special_tokens=False)[0]
    return ids["true"], ids["false"]
 
TRUE_ID, FALSE_ID = get_bool_token_ids(tokenizer)
print(f"True token id: {TRUE_ID}  ({tokenizer.decode([TRUE_ID])})")
print(f"False token id: {FALSE_ID}  ({tokenizer.decode([FALSE_ID])})")
 
# =========================
# LOAD DATA (FORCE START FRESH)
# =========================
print("Loading data...")
 
df = pd.read_excel(excel_file)
 
for col in [column_name, title_col, abstract_col]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
 
start_idx = 0
print("Starting from scratch: row 0")
 
# =========================
# RAG CONTEXT
# =========================
def build_context(row):
    title = str(row[title_col]).strip()
    abstract = str(row[abstract_col]).strip()
 
    if abstract and abstract.lower() != "nan":
        return f"TITLE: {title}\nABSTRACT: {abstract}"
    elif title and title.lower() != "nan":
        return f"TITLE: {title}"
    else:
        return "NO CONTEXT AVAILABLE"
 
# =========================
# DEMOGRAPHIC TOKEN IDS
# =========================
 
def get_first_single_token(tokenizer, variants):
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            return toks[0]
    return tokenizer.encode(variants[0], add_special_tokens=False)[0]
 
GENDER_IDS = {
    "M": get_first_single_token(tokenizer, [" M", "M"]),
    "F": get_first_single_token(tokenizer, [" F", "F"]),
}
 
RACE_IDS = {
    "White":    get_first_single_token(tokenizer, [" White",    "White"]),
    "Black":    get_first_single_token(tokenizer, [" Black",    "Black"]),
    "Asian":    get_first_single_token(tokenizer, [" Asian",    "Asian"]),
    "Hispanic": get_first_single_token(tokenizer, [" Hispanic", "Hispanic"]),
    "Other":    get_first_single_token(tokenizer, [" Other",    "Other"]),
}
 
BRACKET_IDS = {
    "<25":   tokenizer.encode(" <25", add_special_tokens=False)[0],
    "25-50": tokenizer.encode(" 25",  add_special_tokens=False)[0],
    "50+":   tokenizer.encode(" 50",  add_special_tokens=False)[0],
}
 
DIGIT_IDS = {
    str(d): get_first_single_token(tokenizer, [f" {d}", str(d)])
    for d in range(10)
}
 
print("Demographic token ids loaded.")
 
# =========================
# DEMOGRAPHIC LOGPROB SCORERS
# =========================
 
def score_categorical(prompt, token_id_map):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    ids    = list(token_id_map.values())
    labels = list(token_id_map.keys())
    probs  = torch.softmax(torch.stack([logits[i] for i in ids]).float(), dim=0).cpu().numpy()
    best   = int(probs.argmax())
    return labels[best], round(float(probs[best]), 4)
 
 
def score_age(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    digit_ids   = [DIGIT_IDS[str(d)] for d in range(10)]
    digit_probs = torch.softmax(torch.stack([logits[i] for i in digit_ids]).float(), dim=0).cpu().numpy()
    best_digit  = int(digit_probs.argmax())
    confidence  = round(float(digit_probs[best_digit]), 4)
    # Greedy decode to get full age number
    output    = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    age_match = re.match(r"(\d+)", generated)
    age = int(age_match.group(1)) if age_match else best_digit * 10
    return age, confidence
 
 
def score_demographics(quote, context):
    base = f"CONTEXT:\n{context}\n\nQUOTE:\n{quote}\n\n"
    gender,  gender_conf  = score_categorical(base + "GENDER:",  GENDER_IDS)
    race,    race_conf    = score_categorical(base + "RACE:",    RACE_IDS)
    bracket, bracket_conf = score_categorical(base + "BRACKET:", BRACKET_IDS)
    age,     age_conf     = score_age(base + "AGE:")
    return {
        "age": age,           "age_conf":     age_conf,
        "bracket": bracket,   "bracket_conf": bracket_conf,
        "gender": gender,     "gender_conf":  gender_conf,
        "race": race,         "race_conf":    race_conf,
    }
 
# =========================
# IPV PROMPTS
# One prompt per indicator, ending with "KEY:" so the very next token
# the model would generate is the True/False decision. We read that
# directly from logits without generating anything.
# =========================
 
def build_ipv_prompt(indicator_col, quote, context):
    return (
        f"Mark True or False for the following IPV indicator.\n\n"
        f"CONTEXT:\n{context}\n\nQUOTE:\n{quote}\n\n"
        f"{indicator_col.upper()}:"
    )
 
# =========================
# LOG PROB SCORER
# =========================
def score_true_false(prompt: str) -> dict:
    """
    Single forward pass — no generation.
    Reads the logits at the final token position, slices out
    True and False, softmaxes over just those two, returns
    the prediction and calibrated confidence score.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
 
    with torch.no_grad():
        outputs = model(**inputs)
 
    # logits shape: (1, seq_len, vocab_size) — take the last position
    next_token_logits = outputs.logits[0, -1, :]
 
    bool_logits = torch.stack([
        next_token_logits[TRUE_ID],
        next_token_logits[FALSE_ID],
    ])
    bool_probs = torch.softmax(bool_logits.float(), dim=0).cpu().numpy()
 
    p_true  = float(bool_probs[0])
    p_false = float(bool_probs[1])
 
    prediction = "True" if p_true >= p_false else "False"
    confidence = p_true if prediction == "True" else p_false
 
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "p_true":     round(p_true, 4),
        "p_false":    round(p_false, 4),
    }
 
def score_all_ipv(quote: str, context: str) -> dict:
    """Run score_true_false for every indicator, return flat dict."""
    results = {}
    for col_name in IPV_INDICATORS:
        prompt = build_ipv_prompt(col_name, quote, context)
        s = score_true_false(prompt)
        results[col_name]                 = s["prediction"] == "True"
        results[f"{col_name}_confidence"] = s["confidence"]
        results[f"{col_name}_p_true"]     = s["p_true"]
        results[f"{col_name}_p_false"]    = s["p_false"]
    return results
 
# =========================
# RUN LOOP
# =========================
print("Running predictions...")
 
batch_size = 8
 
for i in range(start_idx, len(df), batch_size):
    batch_df = df.iloc[i:i+batch_size]
 
    for j, (_, row) in enumerate(batch_df.iterrows()):
        row_idx = i + j
        ctx = build_context(row)
 
        demo = score_demographics(row[column_name], ctx)
        ipv  = score_all_ipv(row[column_name], ctx)
 
        print(f"\n--- Row {row_idx} ---")
        print("DEMO:", demo)
        print("IPV:", {k: v for k, v in ipv.items() if "_confidence" not in k})
 
        # Demographics
        df.loc[row_idx, "Predicted_Age"]      = demo["age"]
        df.loc[row_idx, "Age_confidence"]     = demo["age_conf"]
        df.loc[row_idx, "Age_Bracket"]        = demo["bracket"]
        df.loc[row_idx, "Bracket_confidence"] = demo["bracket_conf"]
        df.loc[row_idx, "Gender"]             = demo["gender"]
        df.loc[row_idx, "Gender_confidence"]  = demo["gender_conf"]
        df.loc[row_idx, "Race"]               = demo["race"]
        df.loc[row_idx, "Race_confidence"]    = demo["race_conf"]
 
        # IPV indicators + confidence
        for col_name, value in ipv.items():
            df.loc[row_idx, col_name] = value
 
        # Overall confidence — average across all 14 scores
        demo_confs = [demo["age_conf"], demo["bracket_conf"], demo["gender_conf"], demo["race_conf"]]
        ipv_confs  = [v for k, v in ipv.items() if k.endswith("_confidence")]
        df.loc[row_idx, "Overall_confidence"] = round(
            sum(demo_confs + ipv_confs) / len(demo_confs + ipv_confs), 4
        )
 
    # Save progress after every batch
    end_idx = i + len(batch_df)
    df.to_excel(output_progress, index=False)
    print(f"Saved progress: {end_idx}/{len(df)}")
 
# =========================
# FINAL SAVE
# =========================
final_output = excel_file.replace(".xlsx", "_RAG_demographics_restart.xlsx")
df.to_excel(final_output, index=False)
print(f"\nDone! Final file saved to: {final_output}")
 
