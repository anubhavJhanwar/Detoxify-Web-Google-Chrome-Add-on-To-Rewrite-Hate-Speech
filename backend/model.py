"""
model.py
--------
Interpretable toxicity classifier using Logistic Regression.

WHY Logistic Regression:
- Fully interpretable: each feature has a signed weight
- Outputs calibrated probabilities (toxicity score 0–1)
- No black-box: we can directly read which features drive predictions
- Regularization (C parameter) prevents overfitting
- Fast to train and inference

Training data: Jigsaw Toxic Comment Classification Dataset
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score,
)
from sklearn.preprocessing import MaxAbsScaler

from features import FeatureExtractor

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression.pkl"
EXTRACTOR_PATH = MODEL_DIR / "feature_extractor.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jigsaw(csv_path: str, sample_size: int = None) -> tuple[list[str], list[int]]:
    """
    Load Jigsaw Toxic Comment dataset.
    Columns expected: 'comment_text', 'toxic' (binary 0/1).
    If multi-label columns exist, we derive binary toxic label.
    """
    df = pd.read_csv(csv_path)

    # Handle both binary and multi-label Jigsaw formats
    if "toxic" in df.columns:
        label_col = "toxic"
    else:
        # Multi-label: any positive label → toxic
        label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        available = [c for c in label_cols if c in df.columns]
        df["toxic"] = (df[available].sum(axis=1) > 0).astype(int)
        label_col = "toxic"

    text_col = "comment_text" if "comment_text" in df.columns else df.columns[0]

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    # Balance classes to avoid bias (undersample majority)
    toxic = df[df["label"] == 1]
    clean = df[df["label"] == 0]
    min_size = min(len(toxic), len(clean))
    df_balanced = pd.concat([
        toxic.sample(min_size, random_state=42),
        clean.sample(min_size, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    if sample_size:
        df_balanced = df_balanced.head(sample_size)

    return df_balanced["text"].tolist(), df_balanced["label"].tolist()


def load_small_demo_data() -> tuple[list[str], list[int]]:
    """
    Minimal demo dataset for testing when Jigsaw CSV is not available.
    NOT for production — replace with real Jigsaw data.
    """
    toxic_texts = [
        "You are so stupid and worthless!",
        "I hate you, go die in a hole!",
        "You're a complete idiot, shut up!",
        "Nobody likes you, you pathetic loser!",
        "Go to hell you disgusting freak!!",
        "You are trash and everyone knows it",
        "Kill yourself you worthless piece of garbage",
        "I hope you suffer you moron",
        "You're dumb as a rock, get lost",
        "Shut your mouth you ugly idiot!!!",
    ]
    clean_texts = [
        "Have a wonderful day, hope you feel better.",
        "I disagree with your point, but I respect your view.",
        "This is a great idea, thanks for sharing!",
        "Could you please clarify what you mean?",
        "I think there might be a misunderstanding here.",
        "Let's work together to find a solution.",
        "Thank you for your feedback, I appreciate it.",
        "I understand your frustration, let me help.",
        "That's an interesting perspective, tell me more.",
        "I hope we can resolve this peacefully.",
    ]
    texts = toxic_texts + clean_texts
    labels = [1] * len(toxic_texts) + [0] * len(clean_texts)
    return texts, labels


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

class ToxicityClassifier:
    """
    Logistic Regression toxicity classifier with manual feature engineering.
    Fully interpretable: weights are directly readable.
    """

    def __init__(self, max_tfidf_features: int = 10000, C: float = 1.0):
        self.extractor = FeatureExtractor(max_tfidf_features=max_tfidf_features)
        self.scaler = MaxAbsScaler()   # scales sparse matrices without centering
        self.model = LogisticRegression(
            C=C,                        # inverse regularization strength
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",    # handles class imbalance
            random_state=42,
        )
        self.feature_names_: list[str] = []
        self._trained = False

    def fit(self, texts: list[str], labels: list[int], eval: bool = True) -> dict:
        """
        Train on texts/labels with 80/20 stratified split.
        Returns evaluation metrics dict.
        """
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts, labels,
            test_size=0.2,
            stratify=labels,
            random_state=42,
        )

        print(f"Training samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

        # Feature extraction
        print("Extracting features...")
        X_train = self.extractor.fit_transform(X_train_raw)
        X_test = self.extractor.transform(X_test_raw)

        # Scale features (MaxAbsScaler preserves sparsity)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.feature_names_ = self.extractor.get_feature_names()

        # Train
        print("Training Logistic Regression...")
        self.model.fit(X_train, y_train)
        self._trained = True

        metrics = {}
        if eval:
            metrics = self._evaluate(X_test, y_test, X_train, y_train)

        return metrics

    def _evaluate(self, X_test, y_test, X_train, y_train) -> dict:
        """Compute and print evaluation metrics."""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        print("\n=== Evaluation Metrics ===")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=["Clean", "Toxic"]))

        # Check for overfitting
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        print(f"Train accuracy: {train_acc:.4f} | Test accuracy: {metrics['accuracy']:.4f}")
        gap = train_acc - metrics["accuracy"]
        if gap > 0.1:
            print(f"WARNING: Possible overfitting (gap={gap:.4f}). Consider increasing regularization.")
        else:
            print(f"Overfitting check passed (gap={gap:.4f})")

        return metrics

    def predict(self, text: str) -> dict:
        """
        Predict toxicity for a single text.
        Returns score, label, and raw feature vector.
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call fit() or load().")

        vec, names = self.extractor.transform_single(text)
        vec_scaled = self.scaler.transform(vec)

        prob = self.model.predict_proba(vec_scaled)[0]
        toxicity_score = float(prob[1])
        label = int(self.model.predict(vec_scaled)[0])

        return {
            "toxicity_score": round(toxicity_score, 4),
            "label": label,
            "label_str": "toxic" if label == 1 else "clean",
            "vec": vec_scaled,
            "feature_names": names,
        }

    def get_top_features(self, text: str, top_n: int = 10) -> list[dict]:
        """
        Return top contributing features for a prediction.
        Contribution = feature_value * model_weight (for toxic class).
        This is the core of explainability.
        """
        result = self.predict(text)
        vec = result["vec"]
        names = result["feature_names"]

        # Logistic Regression coefficients for the toxic class (index 1)
        coef = self.model.coef_[0]  # shape: (n_features,)

        # Dense array from sparse
        vec_dense = np.asarray(vec.todense()).flatten()

        # Contribution = value * weight
        contributions = vec_dense * coef

        # Get indices of top absolute contributions
        top_idx = np.argsort(np.abs(contributions))[::-1][:top_n]

        top_features = []
        for idx in top_idx:
            if abs(contributions[idx]) < 1e-6:
                continue
            top_features.append({
                "feature": names[idx] if idx < len(names) else f"feature_{idx}",
                "value": round(float(vec_dense[idx]), 4),
                "weight": round(float(coef[idx]), 4),
                "contribution": round(float(contributions[idx]), 4),
                "direction": "increases toxicity" if contributions[idx] > 0 else "decreases toxicity",
            })

        return top_features

    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(EXTRACTOR_PATH, "wb") as f:
            pickle.dump(self.extractor, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {MODEL_DIR}")

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(EXTRACTOR_PATH, "rb") as f:
            self.extractor = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        self.feature_names_ = self.extractor.get_feature_names()
        self._trained = True
        print("Model loaded successfully.")
        return self


def train_model(
    jigsaw_csv: str = None,
    sample_size: int = 20000,
    max_tfidf_features: int = 10000,
    C: float = 1.0,
) -> ToxicityClassifier:
    """
    Train and save the toxicity classifier.
    Uses Jigsaw CSV if provided, otherwise falls back to demo data.
    """
    if jigsaw_csv and Path(jigsaw_csv).exists():
        print(f"Loading Jigsaw dataset from {jigsaw_csv}...")
        texts, labels = load_jigsaw(jigsaw_csv, sample_size=sample_size)
    else:
        print("Jigsaw CSV not found. Using demo data (replace with real dataset for production).")
        texts, labels = load_small_demo_data()

    print(f"Dataset: {len(texts)} samples, {sum(labels)} toxic, {len(labels)-sum(labels)} clean")

    clf = ToxicityClassifier(max_tfidf_features=max_tfidf_features, C=C)
    metrics = clf.fit(texts, labels)
    clf.save()

    return clf


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    clf = train_model(jigsaw_csv=csv_path)

    # Quick inference test
    test_sentences = [
        "You are so stupid and I hate you!!!",
        "Have a wonderful day, hope you feel better.",
        "Go die you worthless idiot!!",
    ]
    print("\n=== Inference Test ===")
    for s in test_sentences:
        result = clf.predict(s)
        print(f"\nText: {s}")
        print(f"  Score: {result['toxicity_score']} | Label: {result['label_str']}")
        top = clf.get_top_features(s, top_n=5)
        for f in top:
            print(f"  [{f['direction']}] {f['feature']}: {f['contribution']:+.4f}")
