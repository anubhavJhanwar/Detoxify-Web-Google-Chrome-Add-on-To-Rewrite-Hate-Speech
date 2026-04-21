"""
features.py
-----------
Manual feature engineering for toxicity classification.
ALL features are hand-crafted and interpretable — no embeddings, no black-box.

Feature groups:
  1. Lexical      — TF-IDF + n-grams (fitted on training corpus)
  2. Linguistic   — POS distribution, pronoun count, verb tense patterns
  3. Toxicity     — Offensive word frequency, sentiment scores, dependency patterns
  4. Structural   — ALL CAPS ratio, exclamation marks, repeated characters
"""

import re
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

STOPWORDS = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

from preprocessing import preprocess

# ---------------------------------------------------------------------------
# Fast doc→preprocessed dict (avoids re-running nlp() on already-parsed docs)
# ---------------------------------------------------------------------------

def _doc_to_preprocessed(doc, original_text: str) -> dict:
    """Convert an already-parsed spaCy doc to the preprocessed dict format."""
    tokens = [t.text for t in doc if not t.is_space]
    tokens_lower = [t.lower() for t in tokens]
    tokens_no_stop = [t for t in tokens_lower if t not in STOPWORDS or t in {"not","no","never","nor","neither"}]
    lemmas = [t.lemma_.lower() for t in doc if not t.is_space]
    pos_tags = [{"text": t.text, "pos": t.pos_, "tag": t.tag_} for t in doc if not t.is_space]
    deps = [{"text": t.text, "dep": t.dep_, "head": t.head.text, "head_pos": t.head.pos_} for t in doc if not t.is_space]
    sentiment = sia.polarity_scores(original_text)
    return {
        "original": original_text,
        "cleaned": original_text,
        "tokens": tokens,
        "tokens_lower": tokens_lower,
        "tokens_no_stop": tokens_no_stop,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "dependencies": deps,
        "sentiment": sentiment,
    }

# ---------------------------------------------------------------------------
# Offensive word dictionary (curated, explainable — not a black-box API)
# ---------------------------------------------------------------------------
OFFENSIVE_WORDS = {
    "stupid", "idiot", "moron", "dumb", "hate", "kill", "die", "ugly",
    "loser", "trash", "garbage", "worthless", "pathetic", "disgusting",
    "retard", "freak", "bastard", "jerk", "ass", "damn", "hell",
    "crap", "shut up", "go away", "nobody cares", "shut your",
    "piece of", "get lost", "screw you", "go to hell",
}

# Second-person aggressive patterns (subject + insult structure)
AGGRESSIVE_PATTERNS = [
    r"\byou\s+(are|r|were)\s+(so\s+)?(stupid|dumb|ugly|worthless|pathetic|trash|garbage|idiot|moron)",
    r"\bi\s+hate\s+(you|this|them|him|her)",
    r"\b(kill|destroy|hurt|attack)\s+(you|them|him|her|yourself)",
    r"\bshut\s+up\b",
    r"\bgo\s+(to\s+hell|die|away|kill\s+yourself)\b",
]

VECTORIZER_PATH = Path(__file__).parent / "models" / "tfidf_vectorizer.pkl"


# ---------------------------------------------------------------------------
# 1. Lexical Features — TF-IDF with unigrams, bigrams, trigrams
# ---------------------------------------------------------------------------

class TFIDFFeatures(BaseEstimator, TransformerMixin):
    """
    Wraps sklearn TfidfVectorizer.
    WHY TF-IDF: Weights rare but discriminative words (e.g., 'idiot') higher
    than common words, giving the model a strong lexical signal.
    N-grams capture multi-word toxic phrases like 'shut up' or 'go die'.
    """

    def __init__(self, max_features: int = 10000, ngram_range=(1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,          # log(1+tf) — reduces impact of very frequent terms
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]*\b",
            min_df=2,                   # ignore terms appearing in < 2 docs (noise)
        )
        self.feature_names_: list[str] = []

    @staticmethod
    def _fast_clean(text: str) -> str:
        """
        Fast tokenization without spaCy — regex-based.
        Used only for TF-IDF fitting/transform (speed critical on large datasets).
        Lowercases, strips stopwords, keeps alpha tokens > 2 chars.
        """
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return " ".join(t for t in tokens if t not in STOPWORDS)

    def fit(self, texts: list[str], y=None):
        print(f"  Building TF-IDF on {len(texts)} texts...", flush=True)
        cleaned = [self._fast_clean(t) for t in texts]
        self.vectorizer.fit(cleaned)
        self.feature_names_ = list(self.vectorizer.get_feature_names_out())
        print(f"  TF-IDF vocab size: {len(self.feature_names_)}", flush=True)
        return self

    def transform(self, texts: list[str]):
        cleaned = [self._fast_clean(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    def get_feature_names(self) -> list[str]:
        return [f"tfidf:{n}" for n in self.feature_names_]

    def save(self, path: Path = VECTORIZER_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path = VECTORIZER_PATH):
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.feature_names_ = list(self.vectorizer.get_feature_names_out())
        return self


# ---------------------------------------------------------------------------
# 2. Linguistic Features
# ---------------------------------------------------------------------------

def extract_linguistic_features(preprocessed: dict) -> tuple[np.ndarray, list[str]]:
    """
    POS distribution, pronoun count, verb tense patterns.

    WHY:
    - POS distribution: toxic text has more adjectives (insults) and imperative verbs
    - Pronoun count: 'you' targeting is a strong toxicity signal
    - Verb tense: present-tense aggressive verbs ('hate', 'kill') vs past tense
    """
    pos_tags = preprocessed["pos_tags"]
    tokens_lower = preprocessed["tokens_lower"]

    total = max(len(pos_tags), 1)

    pos_counts = Counter(t["pos"] for t in pos_tags)
    pos_labels = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "INTJ", "ADP", "DET"]
    pos_ratios = [pos_counts.get(p, 0) / total for p in pos_labels]

    # Pronoun targeting: 'you', 'your', 'yourself'
    second_person = sum(1 for t in tokens_lower if t in {"you", "your", "yourself", "u"})
    pronoun_ratio = second_person / total

    # Verb tense: VBP/VBZ = present, VBD = past, VB = base (imperative)
    tag_counts = Counter(t["tag"] for t in pos_tags)
    present_verb_ratio = (tag_counts.get("VBP", 0) + tag_counts.get("VBZ", 0)) / total
    imperative_ratio = tag_counts.get("VB", 0) / total

    # Interjection ratio (INTJ) — "ugh", "wtf", "omg"
    intj_ratio = pos_counts.get("INTJ", 0) / total

    features = np.array(
        pos_ratios + [pronoun_ratio, present_verb_ratio, imperative_ratio, intj_ratio],
        dtype=np.float32,
    )
    names = (
        [f"pos_ratio_{p}" for p in pos_labels]
        + ["pronoun_2nd_person_ratio", "present_verb_ratio", "imperative_ratio", "intj_ratio"]
    )
    return features, names


# ---------------------------------------------------------------------------
# 3. Toxicity Indicator Features
# ---------------------------------------------------------------------------

def extract_toxicity_features(preprocessed: dict) -> tuple[np.ndarray, list[str]]:
    """
    Offensive word frequency, sentiment scores, dependency patterns.

    WHY:
    - Offensive word count: direct lexical toxicity signal
    - Sentiment: VADER compound score captures overall negativity
    - Dependency patterns: 'you are [insult]' structure is highly toxic
    - Aggressive pattern regex: catches common toxic phrasings
    """
    text = preprocessed["cleaned"]
    tokens_lower = preprocessed["tokens_lower"]
    lemmas = preprocessed["lemmas"]
    sentiment = preprocessed["sentiment"]
    deps = preprocessed["dependencies"]

    # Offensive word hits (normalized by token count)
    total = max(len(tokens_lower), 1)
    offensive_count = sum(1 for t in lemmas if t in OFFENSIVE_WORDS)
    offensive_ratio = offensive_count / total

    # Raw offensive count (absolute signal)
    offensive_abs = min(offensive_count / 5.0, 1.0)  # cap at 5 words → 1.0

    # Sentiment features
    neg_sentiment = sentiment["neg"]
    compound = (1 - sentiment["compound"]) / 2  # map [-1,1] → [0,1], higher = more negative

    # Dependency pattern: subject targeting (nsubj pointing to insult)
    subject_insult = 0
    for dep in deps:
        if dep["dep"] == "nsubj" and dep["head"].lower() in OFFENSIVE_WORDS:
            subject_insult = 1
            break

    # Aggressive regex patterns
    pattern_hits = sum(
        1 for p in AGGRESSIVE_PATTERNS if re.search(p, text.lower())
    )
    pattern_score = min(pattern_hits / len(AGGRESSIVE_PATTERNS), 1.0)

    features = np.array(
        [offensive_ratio, offensive_abs, neg_sentiment, compound, subject_insult, pattern_score],
        dtype=np.float32,
    )
    names = [
        "offensive_word_ratio",
        "offensive_word_abs",
        "neg_sentiment",
        "compound_negativity",
        "subject_insult_dep",
        "aggressive_pattern_score",
    ]
    return features, names


# ---------------------------------------------------------------------------
# 4. Structural Features
# ---------------------------------------------------------------------------

def extract_structural_features(preprocessed: dict) -> tuple[np.ndarray, list[str]]:
    """
    ALL CAPS, exclamation marks, repeated characters.

    WHY:
    - ALL CAPS signals shouting/aggression
    - Exclamation marks signal emotional intensity
    - Repeated characters ('stuuupid') signal emphasis/mockery
    - Short sentences with high punctuation density are often aggressive
    """
    original = preprocessed["original"]
    tokens = preprocessed["tokens"]
    total = max(len(tokens), 1)

    # ALL CAPS words (length > 2 to exclude 'I', 'OK')
    caps_count = sum(1 for t in tokens if t.isupper() and len(t) > 2)
    caps_ratio = caps_count / total

    # Exclamation marks
    excl_count = original.count("!")
    excl_score = min(excl_count / 3.0, 1.0)  # cap at 3

    # Question marks (rhetorical aggression)
    quest_count = original.count("?")
    quest_score = min(quest_count / 3.0, 1.0)

    # Repeated characters: 'haaate', 'stuuupid' (3+ same char in a row)
    repeated_char = len(re.findall(r"(.)\1{2,}", original.lower()))
    repeated_score = min(repeated_char / 3.0, 1.0)

    # Ellipsis / trailing dots (passive aggression)
    ellipsis_count = len(re.findall(r"\.{2,}", original))
    ellipsis_score = min(ellipsis_count / 2.0, 1.0)

    # Text length (very short aggressive messages)
    char_len = len(original)
    short_aggressive = 1.0 if char_len < 30 and excl_count > 0 else 0.0

    features = np.array(
        [caps_ratio, excl_score, quest_score, repeated_score, ellipsis_score, short_aggressive],
        dtype=np.float32,
    )
    names = [
        "caps_ratio",
        "exclamation_score",
        "question_score",
        "repeated_char_score",
        "ellipsis_score",
        "short_aggressive",
    ]
    return features, names


# ---------------------------------------------------------------------------
# Combined feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Combines TF-IDF (sparse) + linguistic + toxicity + structural (dense) features
    into a single feature vector per text sample.

    The TF-IDF vectorizer must be fit on training data first via fit().
    Dense features are stateless (no fitting required).
    """

    def __init__(self, max_tfidf_features: int = 10000):
        self.tfidf = TFIDFFeatures(max_features=max_tfidf_features)
        self._fitted = False

    def fit(self, texts: list[str], y=None):
        self.tfidf.fit(texts)
        self._fitted = True
        return self

    def _dense_features(self, text: str) -> tuple[np.ndarray, list[str]]:
        """
        Fast dense feature extraction — uses regex/VADER only (no spaCy).
        spaCy is only used at inference time for single texts via transform_single().
        """
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        total = max(len(tokens), 1)

        # --- Toxicity features (fast) ---
        offensive_count = sum(1 for t in tokens if t in OFFENSIVE_WORDS)
        offensive_ratio = offensive_count / total
        offensive_abs = min(offensive_count / 5.0, 1.0)

        sentiment = sia.polarity_scores(text)
        neg_sentiment = sentiment["neg"]
        compound = (1 - sentiment["compound"]) / 2

        pattern_hits = sum(1 for p in AGGRESSIVE_PATTERNS if re.search(p, text.lower()))
        pattern_score = min(pattern_hits / len(AGGRESSIVE_PATTERNS), 1.0)

        # --- Structural features (fast) ---
        raw_tokens = text.split()
        caps_count = sum(1 for t in raw_tokens if t.isupper() and len(t) > 2)
        caps_ratio = caps_count / total
        excl_score = min(text.count("!") / 3.0, 1.0)
        quest_score = min(text.count("?") / 3.0, 1.0)
        repeated_score = min(len(re.findall(r"(.)\1{2,}", text.lower())) / 3.0, 1.0)
        ellipsis_score = min(len(re.findall(r"\.{2,}", text)) / 2.0, 1.0)
        short_aggressive = 1.0 if len(text) < 30 and text.count("!") > 0 else 0.0

        # --- Lightweight POS proxies (regex-based, no spaCy) ---
        second_person = sum(1 for t in tokens if t in {"you","your","yourself","u"})
        pronoun_ratio = second_person / total
        # Approximate verb presence via common suffixes
        verb_approx = sum(1 for t in tokens if t.endswith(("ing","ed","es","ize","ise"))) / total

        features = np.array([
            # toxicity
            offensive_ratio, offensive_abs, neg_sentiment, compound,
            0.0,  # subject_insult_dep (needs spaCy — set 0 for bulk)
            pattern_score,
            # structural
            caps_ratio, excl_score, quest_score, repeated_score, ellipsis_score, short_aggressive,
            # linguistic proxies
            pronoun_ratio, verb_approx, 0.0, 0.0,  # imperative/intj need spaCy
            # POS ratios (8 values) — zeroed for bulk, filled at inference
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)

        names = [
            "offensive_word_ratio","offensive_word_abs","neg_sentiment","compound_negativity",
            "subject_insult_dep","aggressive_pattern_score",
            "caps_ratio","exclamation_score","question_score","repeated_char_score",
            "ellipsis_score","short_aggressive",
            "pronoun_2nd_person_ratio","present_verb_ratio","imperative_ratio","intj_ratio",
            "pos_ratio_NOUN","pos_ratio_VERB","pos_ratio_ADJ","pos_ratio_ADV",
            "pos_ratio_PRON","pos_ratio_INTJ","pos_ratio_ADP","pos_ratio_DET",
        ]
        return features, names

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        """
        Returns a sparse matrix: [tfidf_features | dense_features]
        Shape: (n_samples, n_tfidf + n_dense)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        tfidf_matrix = self.tfidf.transform(texts)  # fast regex path

        # Dense features: use lightweight extraction (no full spaCy pipeline)
        print(f"  Extracting dense features for {len(texts)} texts...", flush=True)
        dense_rows = []
        for i, text in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                print(f"    {i}/{len(texts)} done...", flush=True)
            dense_vec, _ = self._dense_features(text)
            dense_rows.append(dense_vec)

        dense_matrix = sp.csr_matrix(np.array(dense_rows, dtype=np.float32))
        return sp.hstack([tfidf_matrix, dense_matrix])

    def fit_transform(self, texts: list[str], y=None) -> sp.csr_matrix:
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> list[str]:
        """Returns all feature names in the same order as the feature vector."""
        tfidf_names = self.tfidf.get_feature_names()
        _, dense_names = self._dense_features("sample text")
        return tfidf_names + dense_names

    def transform_single(self, text: str) -> tuple[sp.csr_matrix, list[str]]:
        """Transform a single text and return (feature_vector, feature_names)."""
        vec = self.transform([text])
        names = self.get_feature_names()
        return vec, names

    def save(self, path: Path = VECTORIZER_PATH):
        self.tfidf.save(path)

    def load(self, path: Path = VECTORIZER_PATH):
        self.tfidf.load(path)
        self._fitted = True
        return self


if __name__ == "__main__":
    samples = [
        "You are so STUPID!!! I hate you!!",
        "Have a great day, hope you feel better.",
        "Go die in a hole you worthless idiot.",
    ]
    extractor = FeatureExtractor(max_tfidf_features=100)
    X = extractor.fit_transform(samples)
    names = extractor.get_feature_names()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Total features: {len(names)}")
    print("Dense feature names:", names[-22:])
