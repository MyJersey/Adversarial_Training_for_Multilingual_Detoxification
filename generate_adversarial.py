import pandas as pd
import torch
import random
import os
import argparse
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
from peft import PeftModel
from tqdm import tqdm

# ========== CONFIG ========== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_RETURN_SEQUENCES = 3
TOP_P = 0.9
MAX_LENGTH = 64
RETRY_TIMES = 2
NONTOXIC_THRESHOLD = 0.5

# ========== CLI argument ========== #
parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, required=True, help="Specify round number to load adapter")
args = parser.parse_args()
round_id = args.round

ADAPTER_PATH = f"models/lora_mt0_round{round_id}"
print(f"Loading LoRA adapter from: {ADAPTER_PATH}")

# ========== Load Validation Data ========== #
df = pd.read_csv("data/lang_val.csv")
prompt_templates = [
    "Please detoxify the following sentence in [{lang}]: ",
    "Make this sentence less toxic in [{lang}]: ",
    "Rewrite in a more polite way in [{lang}]: ",
    "Clean the toxic tone in this sentence in [{lang}]: ",
    "Convert this to a non-toxic version in [{lang}]: ",
]
df["prompt_input"] = [
    random.choice(prompt_templates).format(lang=row["lang"]) + row["toxic"]
    for _, row in df.iterrows()
]

# ========== Load Model & Tokenizer ========== #
base_model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-base").to(DEVICE)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# ========== Load Toxicity Classifier ========== #
clf_tokenizer = AutoTokenizer.from_pretrained("textdetox/xlmr-large-toxicity-classifier-v2", use_fast=False)
clf_model = AutoModelForSequenceClassification.from_pretrained("textdetox/xlmr-large-toxicity-classifier-v2").to(DEVICE)

def get_nontoxic_scores(texts):
    inputs = clf_tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        logits = clf_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    return probs[:, 0].cpu().numpy()  # Index 0 = non-toxic prob

# ========== Generation + Adversarial Detection ========== #
adversarial_examples = []

def rerank_batch(prompts, langs, original_texts):
    all_outputs = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        batch_langs = langs[i:i+BATCH_SIZE]
        batch_toxic = original_texts[i:i+BATCH_SIZE]

        def generate_group(batch):
            encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_length=MAX_LENGTH,
                    do_sample=True,
                    top_p=TOP_P,
                    num_return_sequences=NUM_RETURN_SEQUENCES
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return [decoded[j:j+NUM_RETURN_SEQUENCES] for j in range(0, len(decoded), NUM_RETURN_SEQUENCES)]

        groups = generate_group(batch_prompts)

        for j, candidates in enumerate(groups):
            scores = get_nontoxic_scores(candidates)
            if (scores >= NONTOXIC_THRESHOLD).any():
                chosen = candidates[scores.argmax()]
                all_outputs.append(chosen)
            else:
                retried = False
                for _ in range(RETRY_TIMES):
                    retry_group = generate_group([batch_prompts[j]])[0]
                    retry_scores = get_nontoxic_scores(retry_group)
                    if (retry_scores >= NONTOXIC_THRESHOLD).any():
                        chosen = retry_group[retry_scores.argmax()]
                        all_outputs.append(chosen)
                        retried = True
                        break
                if not retried:
                    adversarial_examples.append({
                        "lang": batch_langs[j],
                        "toxic": batch_toxic[j],
                        "label": candidates[scores.argmax()],
                        "used": False
                    })
                    all_outputs.append(candidates[scores.argmax()])
    return all_outputs

# ========== Run ========== #
print("Generating detoxified outputs & collecting adversarial examples...")
df["detoxified_pred"] = rerank_batch(
    df["prompt_input"].tolist(),
    df["lang"].tolist(),
    df["toxic"].tolist()
)

# Save detoxified output
os.makedirs("data", exist_ok=True)
df[["lang", "toxic", "detoxified_pred"]].to_csv("data/detoxified_val_output.csv", index=False, encoding="utf-8-sig")
print("Detoxified predictions saved to data/detoxified_val_output.csv")

# Save adversarial examples
if adversarial_examples:
    new_adv_df = pd.DataFrame(adversarial_examples)
    adv_file = "data/lang_adversarial.csv"
    if os.path.exists(adv_file):
        existing = pd.read_csv(adv_file)
        adv_df = pd.concat([existing, new_adv_df], ignore_index=True)
        adv_df = adv_df.drop_duplicates(subset=["lang", "toxic", "label"])
    else:
        adv_df = new_adv_df
    adv_df[["lang", "toxic", "label", "used"]].to_csv(adv_file, index=False, encoding="utf-8-sig")
    print(f"Added {len(new_adv_df)} adversarial examples to data/lang_adversarial.csv")
else:
    print("No adversarial examples detected.")
