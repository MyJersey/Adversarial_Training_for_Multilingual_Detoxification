import os
import argparse
import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TOP_P = 0.9
MAX_LENGTH = 64
NUM_RETURN_SEQUENCES = 1

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Path to LoRA adapter model")
parser.add_argument("--output", type=str, default="data/lang_test.csv", help="Path to save detoxified output")
args = parser.parse_args()


df = pd.read_csv("data/lang_reference.csv")

prompt_templates = [
    "Please detoxify the following sentence in [{lang}]: ",
    "Make this sentence less toxic in [{lang}]: ",
    "Rewrite in a more polite way in [{lang}]: ",
    "Clean the toxic tone in this sentence in [{lang}]: ",
    "Convert this to a non-toxic version in [{lang}]: ",
]
df["prompt_input"] = [
    random.choice(prompt_templates).format(lang=row["lang"]) + row["toxic_sentence"]
    for _, row in df.iterrows()
]

adapter_path = args.model or os.environ.get("EVAL_MODEL")
if adapter_path is None:
    raise ValueError("No model path specified.")

base_model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-base").to(DEVICE)
model = PeftModel.from_pretrained(base_model, adapter_path).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

def generate_outputs(prompts):
    outputs = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            gen_output = model.generate(
                **encoded,
                max_length=MAX_LENGTH,
                do_sample=True,
                top_p=TOP_P,
                num_return_sequences=NUM_RETURN_SEQUENCES
            )
        decoded = tokenizer.batch_decode(gen_output, skip_special_tokens=True)
        outputs.extend(decoded)
    return outputs

df["predict"] = generate_outputs(df["prompt_input"].tolist())

df_detox = pd.DataFrame({
    "lang": df["lang"],
    "toxic_sentence": df["toxic_sentence"],
    "neutral_sentence": df["predict"]
})

df_detox.to_csv(args.output, index=False, encoding="utf-8-sig")

print(f"Detoxified output saved to {args.output}")