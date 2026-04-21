"""
explain.py
----------
Explainability module: translates raw model weights into human-readable
explanations for each toxicity prediction.

Approach:
- Logistic Regression coefficients are directly interpretable
- Contribution = feature_value * feature_weight
- We group contributions by category (lexical, linguistic, toxicity, structural)
- We generate natural-language explanations from the top contributors

No LIME, no SHAP — pure weight-based explanation (fully transparent).
"""

from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from preprocessing import preprocess
from features import OFFENSIVE_WORDS, AGGRESSIVE_PATTERNS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    feature: str
    category: str          # lexical | linguistic | toxicity | structural
    value: float
    weight: float
    contribution: float
    human_label: str       # human-readable description
    direction: str         # "increases toxicity" | "decreases toxicity"


@dataclass
class Explanation:
    toxicity_score: float
    label: str
    top_features: list[FeatureContribution]
    summary: str
    toxic_words: list[str]
    aggressive_patterns_found: list[str]
    sentiment_summary: str
    structural_flags: list[str]


# ---------------------------------------------------------------------------
# Category detection helpers
# ---------------------------------------------------------------------------

def _categorize_feature(name: str) -> str:
    if name.startswith("tfidf:"):
        return "lexical"
    if name.startswith("pos_ratio") or name in {
        "pronoun_2nd_person_ratio", "present_verb_ratio",
        "imperative_ratio", "intj_ratio",
    }:
        return "linguistic"
    if name in {
        "offensive_word_ratio", "offensive_word_abs",
        "neg_sentiment", "compound_negativity",
        "subject_insult_dep", "aggressive_pattern_score",
    }:
        return "toxicity"
    if name in {
        "caps_ratio", "exclamation_score", "question_score",
        "repeated_char_score", "ellipsis_score", "short_aggressive",
    }:
        return "structural"
    return "other"


def _human_label(name: str, value: float, contribution: float) -> str:
    """Convert a feature name into a human-readable explanation string."""
    direction = "↑" if contribution > 0 else "↓"

    labels = {
        # Structural
        "caps_ratio": f"{direction} ALL CAPS words ({value*100:.0f}% of tokens)",
        "exclamation_score": f"{direction} Exclamation marks (intensity score {value:.2f})",
        "question_score": f"{direction} Question marks (rhetorical aggression {value:.2f})",
        "repeated_char_score": f"{direction} Repeated characters (e.g., 'stuuupid') score {value:.2f}",
        "ellipsis_score": f"{direction} Trailing dots / ellipsis score {value:.2f}",
        "short_aggressive": f"{direction} Short aggressive message pattern",
        # Toxicity
        "offensive_word_ratio": f"{direction} Offensive word density ({value*100:.1f}% of tokens)",
        "offensive_word_abs": f"{direction} Offensive word count (normalized {value:.2f})",
        "neg_sentiment": f"{direction} Negative sentiment score ({value:.2f})",
        "compound_negativity": f"{direction} Overall negativity (compound {value:.2f})",
        "subject_insult_dep": f"{direction} Subject-insult dependency pattern detected",
        "aggressive_pattern_score": f"{direction} Aggressive phrase pattern score ({value:.2f})",
        # Linguistic
        "pronoun_2nd_person_ratio": f"{direction} Second-person targeting ('you/your') ratio {value:.2f}",
        "present_verb_ratio": f"{direction} Present-tense aggressive verbs ratio {value:.2f}",
        "imperative_ratio": f"{direction} Imperative verb ratio {value:.2f}",
        "intj_ratio": f"{direction} Interjection ratio (e.g., 'wtf', 'ugh') {value:.2f}",
        "pos_ratio_ADJ": f"{direction} Adjective density (insults are often adjectives) {value:.2f}",
        "pos_ratio_VERB": f"{direction} Verb density {value:.2f}",
        "pos_ratio_PRON": f"{direction} Pronoun density {value:.2f}",
        "pos_ratio_INTJ": f"{direction} Interjection density {value:.2f}",
    }

    if name in labels:
        return labels[name]

    if name.startswith("tfidf:"):
        word = name.replace("tfidf:", "")
        return f"{direction} Word/phrase '{word}' (TF-IDF weight {value:.3f})"

    if name.startswith("pos_ratio_"):
        pos = name.replace("pos_ratio_", "")
        return f"{direction} POS ratio for {pos} ({value:.2f})"

    return f"{direction} Feature '{name}' = {value:.3f}"


# ---------------------------------------------------------------------------
# Toxic word detection
# ---------------------------------------------------------------------------

def find_toxic_words(text: str) -> list[str]:
    """Find offensive words present in the text (from our dictionary)."""
    p = preprocess(text)
    lemmas = set(p["lemmas"])
    tokens_lower = set(p["tokens_lower"])
    found = []
    for word in OFFENSIVE_WORDS:
        if " " in word:
            if word in text.lower():
                found.append(word)
        elif word in lemmas or word in tokens_lower:
            found.append(word)
    return sorted(set(found))


def find_aggressive_patterns(text: str) -> list[str]:
    """Return which aggressive regex patterns matched."""
    matched = []
    pattern_labels = [
        "you are [insult]",
        "I hate you/this/them",
        "kill/destroy/hurt [target]",
        "shut up",
        "go to hell / go die",
    ]
    for pattern, label in zip(AGGRESSIVE_PATTERNS, pattern_labels):
        if re.search(pattern, text.lower()):
            matched.append(label)
    return matched


def get_structural_flags(text: str) -> list[str]:
    """Return human-readable structural flags."""
    flags = []
    tokens = text.split()
    caps = [t for t in tokens if t.isupper() and len(t) > 2]
    if caps:
        flags.append(f"ALL CAPS words: {', '.join(caps[:5])}")
    if text.count("!") >= 2:
        flags.append(f"{text.count('!')} exclamation marks")
    if re.search(r"(.)\1{2,}", text.lower()):
        flags.append("Repeated characters detected")
    return flags


# ---------------------------------------------------------------------------
# Main explainer
# ---------------------------------------------------------------------------

class Explainer:
    """
    Wraps a trained ToxicityClassifier and produces human-readable explanations.
    """

    def __init__(self, classifier):
        self.clf = classifier

    def explain(self, text: str, top_n: int = 8) -> Explanation:
        """
        Generate a full explanation for a text prediction.
        """
        result = self.clf.predict(text)
        top_raw = self.clf.get_top_features(text, top_n=top_n)

        # Build FeatureContribution objects
        contributions = []
        for f in top_raw:
            cat = _categorize_feature(f["feature"])
            hl = _human_label(f["feature"], f["value"], f["contribution"])
            contributions.append(FeatureContribution(
                feature=f["feature"],
                category=cat,
                value=f["value"],
                weight=f["weight"],
                contribution=f["contribution"],
                human_label=hl,
                direction=f["direction"],
            ))

        # Supplementary analysis
        toxic_words = find_toxic_words(text)
        patterns = find_aggressive_patterns(text)
        struct_flags = get_structural_flags(text)

        # Sentiment summary
        p = preprocess(text)
        s = p["sentiment"]
        if s["compound"] <= -0.5:
            sent_summary = f"Strongly negative (compound={s['compound']:.2f})"
        elif s["compound"] <= -0.1:
            sent_summary = f"Mildly negative (compound={s['compound']:.2f})"
        elif s["compound"] >= 0.5:
            sent_summary = f"Strongly positive (compound={s['compound']:.2f})"
        else:
            sent_summary = f"Neutral (compound={s['compound']:.2f})"

        # Natural language summary
        summary = self._build_summary(
            result["toxicity_score"],
            result["label_str"],
            toxic_words,
            patterns,
            struct_flags,
            contributions,
        )

        return Explanation(
            toxicity_score=result["toxicity_score"],
            label=result["label_str"],
            top_features=contributions,
            summary=summary,
            toxic_words=toxic_words,
            aggressive_patterns_found=patterns,
            sentiment_summary=sent_summary,
            structural_flags=struct_flags,
        )

    def _build_summary(
        self,
        score: float,
        label: str,
        toxic_words: list[str],
        patterns: list[str],
        struct_flags: list[str],
        contributions: list[FeatureContribution],
    ) -> str:
        parts = []

        if label == "toxic":
            parts.append(f"This text is classified as TOXIC (score: {score:.2f}).")
        else:
            parts.append(f"This text is classified as CLEAN (score: {score:.2f}).")

        if toxic_words:
            parts.append(f"Offensive words detected: {', '.join(toxic_words[:5])}.")

        if patterns:
            parts.append(f"Aggressive patterns found: {'; '.join(patterns)}.")

        if struct_flags:
            parts.append(f"Structural signals: {'; '.join(struct_flags)}.")

        # Top positive contributor
        pos_contribs = [c for c in contributions if c.contribution > 0]
        if pos_contribs:
            top = pos_contribs[0]
            parts.append(f"Strongest toxicity signal: {top.human_label}.")

        return " ".join(parts)

    def to_dict(self, explanation: Explanation) -> dict:
        """Serialize Explanation to a JSON-serializable dict."""
        return {
            "toxicity_score": explanation.toxicity_score,
            "label": explanation.label,
            "summary": explanation.summary,
            "toxic_words": explanation.toxic_words,
            "aggressive_patterns_found": explanation.aggressive_patterns_found,
            "sentiment_summary": explanation.sentiment_summary,
            "structural_flags": explanation.structural_flags,
            "top_features": [
                {
                    "feature": c.feature,
                    "category": c.category,
                    "value": c.value,
                    "weight": c.weight,
                    "contribution": c.contribution,
                    "human_label": c.human_label,
                    "direction": c.direction,
                }
                for c in explanation.top_features
            ],
        }


if __name__ == "__main__":
    from model import ToxicityClassifier, train_model
    from pathlib import Path

    clf = train_model()  # trains on demo data
    explainer = Explainer(clf)

    test = "You are so STUPID!!! I hate you and everyone like you!!"
    exp = explainer.explain(test)

    import json
    print(json.dumps(explainer.to_dict(exp), indent=2))
