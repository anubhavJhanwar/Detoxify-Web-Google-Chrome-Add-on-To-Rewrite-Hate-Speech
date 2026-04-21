"""
preprocessing.py
----------------
Core NLP preprocessing pipeline using classical NLP techniques (spaCy + NLTK).
No transformers, no black-box models.

WHY each step:
- Sentence segmentation: split compound inputs into logical units
- Tokenization: break text into atomic units for analysis
- Lowercasing: normalize surface forms (Dog == dog)
- Stopword removal: reduce noise from high-frequency, low-signal words
- Lemmatization: reduce inflected forms to base (running -> run)
- POS tagging: identify grammatical roles for feature engineering
- Dependency parsing: capture syntactic relationships (subject+verb+object)
"""

import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)

# Load spaCy model (small English model — classical, not transformer)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

STOPWORDS = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()


def clean_text(text: str) -> str:
    """
    Light cleaning: remove URLs, excessive whitespace, non-ASCII junk.
    We intentionally keep punctuation and casing at this stage
    so downstream steps can use them as features.
    """
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)           # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text


def segment_sentences(text: str) -> list[str]:
    """
    Sentence segmentation using spaCy's dependency-based sentencizer.
    WHY: A single input may contain multiple sentences with different tones.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def tokenize(text: str) -> list[str]:
    """
    Tokenization using spaCy's rule-based tokenizer.
    WHY: Handles contractions, punctuation, and special cases correctly.
    """
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]


def lowercase_tokens(tokens: list[str]) -> list[str]:
    """
    Lowercasing.
    WHY: Normalize surface forms so 'HATE' and 'hate' map to the same feature.
    We preserve original tokens separately for structural feature extraction.
    """
    return [t.lower() for t in tokens]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Stopword removal.
    WHY: Words like 'the', 'is', 'at' carry little discriminative signal
    for toxicity classification. Removing them reduces feature space noise.
    NOTE: We keep negations ('not', 'no', 'never') — they flip sentiment.
    """
    negations = {"not", "no", "never", "nor", "neither", "without"}
    return [t for t in tokens if t not in STOPWORDS or t in negations]


def lemmatize(doc) -> list[str]:
    """
    Lemmatization using spaCy's lookup + rule-based lemmatizer.
    WHY: 'hating', 'hated', 'hates' all reduce to 'hate',
    improving feature generalization without losing meaning.
    """
    return [token.lemma_.lower() for token in doc if not token.is_space]


def pos_tag(doc) -> list[dict]:
    """
    POS tagging using spaCy's statistical tagger (trained on OntoNotes).
    WHY: POS distribution is a strong toxicity signal —
    toxic text tends to have more adjectives/verbs of aggression.
    Returns both coarse (pos_) and fine-grained (tag_) tags.
    """
    return [
        {
            "text": token.text,
            "pos": token.pos_,    # coarse: NOUN, VERB, ADJ ...
            "tag": token.tag_,    # fine: NNS, VBZ, JJR ...
        }
        for token in doc
        if not token.is_space
    ]


def dependency_parse(doc) -> list[dict]:
    """
    Dependency parsing using spaCy's arc-eager parser.
    WHY: Captures syntactic relationships like 'nsubj' (nominal subject)
    and 'dobj' (direct object). Useful for detecting patterns like
    'you are [insult]' or '[subject] [aggressive verb] [target]'.
    """
    return [
        {
            "text": token.text,
            "dep": token.dep_,
            "head": token.head.text,
            "head_pos": token.head.pos_,
        }
        for token in doc
        if not token.is_space
    ]


def get_sentiment(text: str) -> dict:
    """
    VADER sentiment analysis (rule-based, lexicon-driven).
    WHY: Provides compound, positive, negative, neutral scores
    as direct features for the classifier. Toxic text skews negative.
    """
    return sia.polarity_scores(text)


def preprocess(text: str) -> dict:
    """
    Full pipeline: runs all steps and returns a structured output dict.

    Returns
    -------
    {
        "original": str,
        "cleaned": str,
        "sentences": [str],
        "tokens": [str],
        "tokens_lower": [str],
        "tokens_no_stop": [str],
        "lemmas": [str],
        "pos_tags": [{"text", "pos", "tag"}],
        "dependencies": [{"text", "dep", "head", "head_pos"}],
        "sentiment": {"neg", "neu", "pos", "compound"},
    }
    """
    cleaned = clean_text(text)
    doc = nlp(cleaned)

    raw_tokens = tokenize(cleaned)
    lower_tokens = lowercase_tokens(raw_tokens)
    no_stop = remove_stopwords(lower_tokens)
    lemmas = lemmatize(doc)
    pos_tags = pos_tag(doc)
    deps = dependency_parse(doc)
    sentences = segment_sentences(cleaned)
    sentiment = get_sentiment(cleaned)

    return {
        "original": text,
        "cleaned": cleaned,
        "sentences": sentences,
        "tokens": raw_tokens,
        "tokens_lower": lower_tokens,
        "tokens_no_stop": no_stop,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "dependencies": deps,
        "sentiment": sentiment,
    }


if __name__ == "__main__":
    sample = "You are so STUPID!!! I hate you and everyone like you!!"
    result = preprocess(sample)
    import json
    print(json.dumps(result, indent=2))
