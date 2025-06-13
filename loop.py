import os
import shutil
import pandas as pd
import subprocess
from datetime import datetime
import json
import re

ADVERSARIAL_PATH = "data/lang_adversarial.csv"
RESULTS_DIR = "results"
ADVERSARIAL_THRESHOLD = 10
STABLE_LIMIT = 2
REFERENCE_PATH = "data/lang_reference.csv"
LANG_TO_EVAL = "uk"

def count_total_adversarial():
    if not os.path.exists(ADVERSARIAL_PATH):
        return 0
    df = pd.read_csv(ADVERSARIAL_PATH)
    return len(df)

def train_round(round_id):
    print(f"Training Round {round_id}")
    os.environ["CURRENT_ROUND"] = str(round_id)
    subprocess.run(["python", "train.py"], check=True)

    src_model_dir = os.path.join("models", "lora_mt0")
    dst_model_dir = os.path.join("models", f"lora_mt0_round{round_id}")

    if os.path.exists(dst_model_dir):
        shutil.rmtree(dst_model_dir)
    shutil.copytree(src_model_dir, dst_model_dir)
    print(f"Saved model snapshot to {dst_model_dir}")

def generate_adversarial(round_id):
    print(f"Generating adversarial examples for Round {round_id}...")
    subprocess.run(["python", "generate_adversarial.py", "--round", str(round_id)], check=True)

def evaluate_model(round_id):
    print(f"Evaluating detox performance for Round {round_id}...")

    adapter_path = f"models/lora_mt0_round{round_id}"
    submission_path = f"data/lang_test_round{round_id}.csv"

    subprocess.run([
        "python", "test.py",
        "--model", adapter_path,
        "--output", submission_path
    ], check=True)

    eval_command = [
        "conda", "run", "-n", "metric",
        "python", "evaluation/evaluate.py",
        "--submission", submission_path,
        "--reference", REFERENCE_PATH,
        "--device", "cuda",
        "--batch_size", "32",
        "--fluency_batch_size", "32",
        "--efficient", "False"
    ]

    try:
        result = subprocess.run(
            eval_command,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)

        # Fallback: parse markdown output table
        lines = result.stdout.splitlines()
        headers = []
        values = []
        for line in lines:
            if "lang" in line and "|" in line:
                headers = [h.strip() for h in line.split("|")[2:] if h.strip()]
            elif headers and "|" in line and not line.strip().startswith("|---"):
                values = [v.strip() for v in line.split("|")[2:] if v.strip()]
                if len(values) == len(headers):
                    break

        if headers and values:
            row_dict = dict(zip(headers, values))
            metrics = [{
                "lang": row_dict.get("lang"),
                "STA": float(row_dict.get("STA", 0)),
                "SIM": float(row_dict.get("SIM", 0)),
                "XCOMET": float(row_dict.get("XCOMET", 0)),
                "J": float(row_dict.get("J", 0))
            }]
            return metrics
        else:
            print("Markdown table parsing failed.")
            return None

    except subprocess.CalledProcessError as e:
        print("Evaluation failed:", e.stderr)
        return None


def save_results(round_id, adv_count, score, tox, sim, flu):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, "train_log.csv")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[round_id, now, adv_count, score, tox, sim, flu]],
                         columns=["round", "timestamp", "adversarial_count", "score",
                                  "toxicity_score", "similarity_score", "fluency_score"])
    if os.path.exists(result_file):
        existing = pd.read_csv(result_file)
        updated = pd.concat([existing, entry], ignore_index=True)
    else:
        updated = entry
    updated.to_csv(result_file, index=False)

def main():
    print("Starting adversarial training loop...")

    prev_total = count_total_adversarial()
    stable_rounds = 0
    round_id = 0
    best_score = -1
    best_round = None

    while True:
        round_id += 1
        print(f"\n=== ROUND {round_id} ===")

        train_round(round_id)
        generate_adversarial(round_id)

        current_total = count_total_adversarial()
        new_added = current_total - prev_total
        prev_total = current_total

        metrics = evaluate_model(round_id)

        score = tox = sim = flu = None

        if isinstance(metrics, list):
            for row in metrics:
                if row.get("lang") == LANG_TO_EVAL:
                    score = row.get("J")
                    tox = row.get("STA")
                    sim = row.get("SIM")
                    flu = row.get("XCOMET")
                    break


        save_results(round_id, current_total, score, tox, sim, flu)

        if score is not None and score > best_score:
            best_score = score
            best_round = round_id
            with open(os.path.join(RESULTS_DIR, "best_model.txt"), "w") as f:
                f.write(f"Best model so far: Round {best_round} with score {best_score:.4f}\n")

        print(f"New adversarial samples: {new_added}, DetoxEval J-score: {score:.4f}" if score is not None else "Evaluation failed.")

        if new_added < ADVERSARIAL_THRESHOLD:
            stable_rounds += 1
            print(f"Low adversarial growth → stable_rounds = {stable_rounds}")
        else:
            stable_rounds = 0

        if stable_rounds >= STABLE_LIMIT:
            print(f"Stopping: adversarial growth plateaued.")
            break

    print("Training loop complete.")
    if best_round:
        print(f"Best model selected: Round {best_round} with score {best_score:.4f}")

if __name__ == "__main__":
    main()