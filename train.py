import os
import random
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch


def load_data():
    df_train = pd.read_csv("data/lang_train.csv")
    df_val = pd.read_csv("data/lang_val.csv")

    prompt_templates = [
        "Please detoxify the following sentence in [{lang}]: ",
        "Make this sentence less toxic in [{lang}]: ",
        "Rewrite in a more polite way in [{lang}]: ",
        "Clean the toxic tone in this sentence in [{lang}]: ",
        "Convert this to a non-toxic version in [{lang}]: ",
    ]

    def apply_prompt(df):
        return [
            random.choice(prompt_templates).format(lang=row["lang"]) + row["toxic"]
            for _, row in df.iterrows()
        ]

    df_train["prompt_input"] = apply_prompt(df_train)
    df_val["prompt_input"] = apply_prompt(df_val)

    adv_path = "data/lang_adversarial.csv"
    if os.path.exists(adv_path):
        df_adv = pd.read_csv(adv_path)
        if "used" not in df_adv.columns:
            df_adv["used"] = False
        unused = df_adv[df_adv["used"] == False]

        if not unused.empty:
            df_new = unused.drop_duplicates(subset=["lang", "toxic", "label"])
            df_adv.loc[df_new.index, "used"] = True
            df_adv.to_csv(adv_path, index=False, encoding="utf-8-sig")

            df_new["prompt_input"] = [
                random.choice(prompt_templates).format(lang=row["lang"]) + row["toxic"]
                for _, row in df_new.iterrows()
            ]
            df_new = df_new.rename(columns={"label": "detoxified"})

            df_train = pd.concat([df_train, df_new[["lang", "toxic", "prompt_input", "detoxified"]]], ignore_index=True)

    train_dataset = Dataset.from_pandas(
        df_train[["prompt_input", "detoxified"]].rename(columns={"prompt_input": "input", "detoxified": "label"})
    )
    val_dataset = Dataset.from_pandas(
        df_val[["prompt_input", "detoxified"]].rename(columns={"prompt_input": "input", "detoxified": "label"})
    )
    return train_dataset, val_dataset


def preprocess(examples, tokenizer):
    model_inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=64)
    labels = tokenizer(text_target=examples["label"], truncation=True, padding="max_length", max_length=64)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train():
    round_id = int(os.environ.get("CURRENT_ROUND", "1"))
    print(f"Current training round: {round_id}")

    if round_id > 1:
        prev_model_path = f"models/lora_mt0_round{round_id - 1}"
    else:
        prev_model_path = "bigscience/mt0-base"

    checkpoint_dir = "models/lora_mt0/checkpoint-last"
    resume = os.path.exists(checkpoint_dir)

    print(f"Loading model from: {prev_model_path}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(prev_model_path, use_safetensors=False)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v", "wo"],
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(base_model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-base", use_safetensors=False)

    train_dataset, val_dataset = load_data()

    tokenized_train = train_dataset.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=val_dataset.column_names)

    args = Seq2SeqTrainingArguments(
        output_dir="models/lora_mt0",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-4,
        num_train_epochs=8,
        gradient_accumulation_steps=1,
        fp16=False,
        logging_steps=50,
        save_total_limit=1,
        evaluation_strategy="epoch",
        predict_with_generate=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="epoch",
        warmup_ratio=0.1,
        weight_decay=0.01,
        generation_max_length=50,
        generation_num_beams=1
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    if resume:
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("Starting training from scratch or previous round...")
        trainer.train()

    print("Saving model to models/lora_mt0")
    model.save_pretrained("models/lora_mt0")
    tokenizer.save_pretrained("models/lora_mt0")


if __name__ == '__main__':
    train()
