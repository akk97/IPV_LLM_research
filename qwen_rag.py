import pandas as pd
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# =========================
# CONFIG
# =========================
model_cache = "Qwen/Qwen2.5-7B-Instruct"
lora_path = "/home/akk97/lora_4tasks_binaryrelevance_adapter_v3_qwen/checkpoint-1041"

excel_file = "/home/akk97/IPVData_cleaned.xlsx"
column_name = "quotations_cleaned"
title_col = "Title of the articles"
abstract_col = "abstracts"

output_progress = "rag_qwen_labels_progress.xlsx"

IPV_INDICATORS = [
    "Mental_Health",
    "Rural",
    "Neighborhood_Safety_Crime",
    "Family",
    "Income",
]

# =========================
# LOAD MODEL
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

model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

print("Model loaded successfully.")

# =========================
# TRUE/FALSE TOKEN IDS
# =========================
def get_bool_token_ids(tokenizer):
    candidates = {
        "true":  [" True", "True", " true", "true"],
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

# =========================
# LOAD DATA
# =========================
print("Loading data...")
df = pd.read_excel(excel_file)

for col in [column_name, title_col, abstract_col]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

start_idx = 0
print("Starting from scratch: row 0")

# =========================
# CONTEXT
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
# TOKEN HELPERS
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
    "White":    get_first_single_token(tokenizer, [" White", "White"]),
    "Black":    get_first_single_token(tokenizer, [" Black", "Black"]),
    "Asian":    get_first_single_token(tokenizer, [" Asian", "Asian"]),
    "Hispanic": get_first_single_token(tokenizer, [" Hispanic", "Hispanic"]),
    "Other":    get_first_single_token(tokenizer, [" Other", "Other"]),
}

# =========================
# AGE SCORING
# =========================
def score_sequence_batch(prompts, labels):
    texts = [p + " " + l for p, l in zip(prompts, labels)]

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs).logits

    log_probs = torch.log_softmax(outputs, dim=-1)
    input_ids = inputs["input_ids"]

    scores = []

    for b in range(len(texts)):
        seq_score = 0.0
        for i in range(1, input_ids.shape[1]):
            token_id = input_ids[b, i].item()
            if token_id == tokenizer.pad_token_id:
                continue
            seq_score += log_probs[b, i-1, token_id].item()
        scores.append(seq_score)

    return scores


def score_age(prompt):
    labels = ["<25", "25-50", "50+"]
    scores = score_sequence_batch([prompt]*len(labels), labels)
    probs = torch.softmax(torch.tensor(scores), dim=0).cpu().numpy()
    best = int(probs.argmax())
    return labels[best], round(float(probs[best]), 4)

# =========================
# DEMOGRAPHICS
# =========================
def score_categorical(prompt, token_id_map):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    ids = list(token_id_map.values())
    labels = list(token_id_map.keys())

    probs = torch.softmax(
        torch.stack([logits[i] for i in ids]).float(), dim=0
    ).cpu().numpy()

    best = int(probs.argmax())
    return labels[best], round(float(probs[best]), 4)


def score_demographics(quote, context):
    base = f"CONTEXT:\n{context}\n\nQUOTE:\n{quote}\n\n"

    gender, gender_conf = score_categorical(base + "GENDER (M/F): ", GENDER_IDS)
    race, race_conf = score_categorical(base + "RACE (White/Black/Asian/Hispanic/Other): ", RACE_IDS)
    bracket, bracket_conf = score_age(base + "AGE BRACKET (<25, 25-50, 50+): ")

    return {
        "gender": gender,
        "gender_conf": gender_conf,
        "race": race,
        "race_conf": race_conf,
        "bracket": bracket,
        "bracket_conf": bracket_conf,
    }

# =========================
#  DEFINITION
# =========================
MENTAL_HEALTH_DEFINITION = """
Mental Health Indicator includes cognitive distortions such as:
- Catastrophizing
- Overgeneralization
- Black-and-white thinking
- Emotional reasoning
- Mind reading
- Labeling
- Personalization
"""

RURAL_DEFINITION = """
Rural refers ONLY to clearly non-urban environments.

Mark True ONLY if the quote explicitly indicates:
- Village, farm, countryside, remote area
- Agricultural setting (farming, livestock, fields)
- Sparse population or isolation

Do NOT infer rural from:
- Poverty
- Small problems or personal struggles
- Lack of services (unless explicitly tied to rural setting)

If the setting is unclear or ambiguous → mark False.

Default to False unless there is explicit evidence.
"""

# =========================
# IPV SCORING
# =========================
def build_ipv_prompt(indicator_col, quote, context):

    definition = ""
    if indicator_col == "Mental_Health":
        definition = MENTAL_HEALTH_DEFINITION
    
    elif indicator_col == "Rural":
        definition = RURAL_DEFINITION

    return (
        f"Mark True or False.\n\n"
        f"{definition}\n\n"
        f"CONTEXT:\n{context}\n\nQUOTE:\n{quote}\n\n"
        f"{indicator_col.upper()}: "
    )


def score_true_false(prompt: str) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    bool_logits = torch.stack([
        logits[TRUE_ID],
        logits[FALSE_ID],
    ])

    probs = torch.softmax(bool_logits.float(), dim=0).cpu().numpy()

    p_true = float(probs[0])
    p_false = float(probs[1])

    prediction = "True" if p_true >= p_false else "False"
    confidence = p_true if prediction == "True" else p_false

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "p_true": round(p_true, 4),
        "p_false": round(p_false, 4),
    }


def score_all_ipv(quote: str, context: str) -> dict:
    results = {}

    for col_name in IPV_INDICATORS:
        prompt = build_ipv_prompt(col_name, quote, context)
        s = score_true_false(prompt)

        results[col_name] = s["prediction"] == "True"
        results[f"{col_name}_confidence"] = s["confidence"]
        results[f"{col_name}_p_true"] = s["p_true"]
        results[f"{col_name}_p_false"] = s["p_false"]

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
        ipv = score_all_ipv(row[column_name], ctx)

        df.loc[row_idx, "Gender"] = demo["gender"]
        df.loc[row_idx, "Gender_conf"] = demo["gender_conf"]

        df.loc[row_idx, "Race"] = demo["race"]
        df.loc[row_idx, "Race_conf"] = demo["race_conf"]

        df.loc[row_idx, "Age_Bracket"] = demo["bracket"]
        df.loc[row_idx, "Age_Bracket_conf"] = demo["bracket_conf"]

        for col_name, value in ipv.items():
            df.loc[row_idx, col_name] = value

    # ATOMIC SAVE FIX
    temp_file = output_progress.replace(".xlsx", ".tmp.xlsx")
    df.to_excel(temp_file, index=False)
    os.replace(temp_file, output_progress)

    print(f"Saved progress: {i + len(batch_df)}")

# =========================
# FINAL SAVE
# =========================
final_output = excel_file.replace(".xlsx", "_RAG_demographics_final.xlsx")
df.to_excel(final_output, index=False)

print("Done:", final_output)