"""
evaluate.py
-----------
Evaluation module for both classification and rewriting quality.

Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC
Rewriting metrics: Semantic similarity (cosine TF-IDF), BLEU score
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

from model import ToxicityClassifier, load_jigsaw, load_small_demo_data
from rewrite import TextRewriter, semantic_similarity


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(clf: ToxicityClassifier, texts: list[str], labels: list[int]) -> dict:
    """
    Evaluate classifier on a held-out test set.
    Returns dict of all metrics.
    """
    predictions = []
    probabilities = []

    for text in texts:
        result = clf.predict(text)
        predictions.append(result["label"])
        probabilities.append(result["toxicity_score"])

    y_pred = np.array(predictions)
    y_prob = np.array(probabilities)
    y_true = np.array(labels)

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    print("\n=== Classification Evaluation ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:12s}: {v}")
    print(f"\n  Confusion Matrix:\n  {cm}")
    print("\n" + classification_report(y_true, y_pred, target_names=["Clean", "Toxic"]))

    return metrics


# ---------------------------------------------------------------------------
# Rewriting evaluation
# ---------------------------------------------------------------------------

def bleu_score_simple(reference: str, hypothesis: str) -> float:
    """
    Simple unigram + bigram BLEU approximation (no NLTK dependency for BLEU).
    WHY: Measures n-gram overlap between rewritten text and reference neutral text.
    """
    from collections import Counter

    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return 0.0

    # Unigram precision
    ref_1 = Counter(ref_tokens)
    hyp_1 = Counter(hyp_tokens)
    match_1 = sum(min(hyp_1[w], ref_1[w]) for w in hyp_1)
    p1 = match_1 / len(hyp_tokens) if hyp_tokens else 0

    # Bigram precision
    ref_2 = Counter(ngrams(ref_tokens, 2))
    hyp_2 = Counter(ngrams(hyp_tokens, 2))
    match_2 = sum(min(hyp_2[b], ref_2[b]) for b in hyp_2)
    p2 = match_2 / max(len(hyp_tokens) - 1, 1)

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))

    bleu = bp * (p1 * p2) ** 0.5
    return round(bleu, 4)


def evaluate_rewriter(
    rewriter: TextRewriter,
    toxic_texts: list[str],
    reference_neutrals: list[str] = None,
) -> dict:
    """
    Evaluate rewriting quality.
    - Semantic similarity: cosine TF-IDF between original and rewritten
    - BLEU: if reference neutrals provided (from ParaDetox)
    """
    similarities = []
    bleu_scores = []
    rewrite_rates = []

    for i, text in enumerate(toxic_texts):
        result = rewriter.rewrite_with_pos(text)
        rewritten = result["rewritten"]

        sim = semantic_similarity(text, rewritten)
        similarities.append(sim)
        rewrite_rates.append(1 if result["toxicity_reduced"] else 0)

        if reference_neutrals and i < len(reference_neutrals):
            bleu = bleu_score_simple(reference_neutrals[i], rewritten)
            bleu_scores.append(bleu)

    metrics = {
        "avg_semantic_similarity": round(np.mean(similarities), 4),
        "min_semantic_similarity": round(np.min(similarities), 4),
        "rewrite_rate": round(np.mean(rewrite_rates), 4),
    }

    if bleu_scores:
        metrics["avg_bleu"] = round(np.mean(bleu_scores), 4)

    print("\n=== Rewriting Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

    return metrics


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(jigsaw_csv: str = None, sample_size: int = 500):
    """Run complete evaluation on both classifier and rewriter."""
    from model import train_model

    print("Loading data...")
    if jigsaw_csv and Path(jigsaw_csv).exists():
        texts, labels = load_jigsaw(jigsaw_csv, sample_size=sample_size)
    else:
        texts, labels = load_small_demo_data()

    print(f"Evaluating on {len(texts)} samples...")

    # Train (or load) model
    clf = ToxicityClassifier()
    model_path = Path(__file__).parent / "models" / "logistic_regression.pkl"
    if model_path.exists():
        clf.load()
    else:
        clf = train_model(jigsaw_csv=jigsaw_csv, sample_size=sample_size)

    # Classification evaluation
    clf_metrics = evaluate_classifier(clf, texts, labels)

    # Rewriting evaluation (on toxic subset)
    rewriter = TextRewriter()
    toxic_texts = [t for t, l in zip(texts, labels) if l == 1][:50]
    rw_metrics = evaluate_rewriter(rewriter, toxic_texts)

    return {"classification": clf_metrics, "rewriting": rw_metrics}


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    results = run_full_evaluation(jigsaw_csv=csv)
    import json
    print("\n=== Final Results ===")
    print(json.dumps(results, indent=2))
