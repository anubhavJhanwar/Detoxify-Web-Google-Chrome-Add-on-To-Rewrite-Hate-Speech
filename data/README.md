# Dataset Instructions

## Primary Dataset — Jigsaw Toxic Comment Classification

**Purpose:** Train and evaluate the toxicity classification model (Logistic Regression).

**Download:**
1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Download `train.csv`
3. Rename it to `jigsaw_dataset.csv` and place it in this `data/` folder.

**Expected columns:**
- `comment_text` — the text
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` — binary labels

The model uses the `toxic` column as the primary binary label.
If only multi-label columns exist, any positive label → toxic=1.

---

## Supporting Dataset — ParaDetox

**Purpose:** Extract toxic→neutral rewriting patterns for the rule-based rewriting engine.
We do NOT train any ML model on this dataset.

**Download:**
1. Go to: https://huggingface.co/datasets/s-nlp/paradetox
   OR: https://github.com/s-nlp/paradetox
2. Download the TSV/CSV file with toxic/neutral pairs
3. Save as `paradetox_pairs.csv` in this `data/` folder.

**Expected columns:**
- `toxic_sentence` — original toxic text
- `neutral_sentence` — human-written neutral rewrite

**How it's used:**
The `rewrite.py` module contains `PARADETOX_MAPPINGS` — a curated list of
toxic→neutral phrase pairs derived from analyzing the ParaDetox dataset.
These are used as lookup rules, not for model training.

---

## Without Datasets

If neither CSV is present, the system falls back to a small built-in demo dataset
(10 toxic + 10 clean examples) for demonstration purposes.
For production accuracy, always use the full Jigsaw dataset.
