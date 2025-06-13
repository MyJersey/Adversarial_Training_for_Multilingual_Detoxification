
# Multilingual Text Detoxification with Adversarial Training

This repository implements a multilingual detoxification pipeline using LoRA-adapted mT0 models. The model is trained to transform toxic input sentences into neutral counterparts across several languages. It applies iterative adversarial fine-tuning, automatically mining hard examples and retraining the model in multiple rounds.

---

## Task Background

This project is based on the **CLEF 2025 Detoxification Shared Task**, which challenges models to detoxify toxic text across languages while preserving meaning and fluency.

> **Dataset**: All detoxification data are obtained from the official PAN-CLEF 2025 dataset ([available on Hugging Face](https://huggingface.co/datasets/textdetox/multilingual_paradetox))
>  

---

## Project Structure

```
.
├── generate_adversarial.py      # Generates detox outputs & collects adversarial examples
├── loop.py                      # Full training loop controller
├── train.py                     # Trains LoRA adapters on detoxification pairs
├── test.py                      # Predicts test set outputs using trained adapters
├── data/
│   ├── lang_train.csv           # Training data
│   ├── lang_val.csv             # Validation data
│   ├── lang_test.csv            # Test predictions
│   ├── lang_adversarial.csv     # Hard examples mined during training
├── models/                      # LoRA adapter checkpoints (per round)
├── results/                     # Evaluation logs and best model record
└── evaluation/                  # Evaluation code (adapted from PAN-CLEF 2025 baseline)
```

---

## Evaluation Method

The evaluation is fully automatic and based on the PAN-CLEF 2025 baseline pipeline. The code in `evaluation/` is adapted from [pan-webis-de/pan-code](https://github.com/pan-webis-de/pan-code), licensed under the MIT License.

### Evaluated Metrics:
- **STA** – Style Transfer Adequacy (toxicity reduction)
- **SIM** – Semantic similarity via LaBSE
- **XCOMET** – Fluency via COMET
- **J** – Composite DetoxEval score

The evaluation script is invoked in each round using `conda run` and parses markdown output to select the best-performing round.

---

## Core Logic

Each round involves:
1. **Training** using LoRA adapters on mT0.
2. **Generation** of detoxified predictions on validation data.
3. **Adversarial Mining** for failed detoxifications.
4. **Evaluation** using automatic metrics.
5. **Loop** until no more adversarial gain is found.

---

## Installation & Environments

We recommend using two separate environments for training and evaluation.

### Detox Training Environment (`conda create -n detox`)

```txt
transformers==4.35.0
sentence-transformers==2.6.1
peft==0.10.0
datasets==3.6.0
torch==2.5.1
accelerate==0.23.0
pandas==2.3.0
numpy==1.26.4
tqdm==4.67.1
protobuf==4.25.6
```

Install with:

```bash
conda create -n detox python=3.10
conda activate detox
pip install -r requirements.txt
```

### Evaluation Environment (`conda create -n metric`)

```txt
sentence-transformers==2.6.1
transformers==4.35.0
unbabel_comet==2.2.1
pandas==2.2.2
pydantic==2.9.2
pydantic-settings==2.6.1
pydantic_core==2.23.4
scipy==1.14.1
huggingface_hub==0.30.2
numpy==1.26.4
protobuf==4.25.6
tqdm==4.66.5
```

Run evaluation using:

```bash
conda activate metric
python evaluation/evaluate.py --submission data/lang_test.csv --reference data/lang_reference.csv --device cuda
```

---

## Output Artifacts

- `data/lang_test_round{n}.csv` — Detoxified predictions
- `results/train_log.csv` — Per-round metrics (toxicity, similarity, fluency)
- `results/best_model.txt` — Best checkpoint summary
- `data/lang_adversarial.csv` — Mined adversarial examples

---


## Maintainer

This repo was built and extended by Jersey for multilingual detoxification research and experimentation at University of Copenhagen

For issues or questions, please open an issue or contact the maintainer.
