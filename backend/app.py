"""
app.py
------
FastAPI backend for the Explainable Toxic Text Normalization system.

Endpoints:
  POST /analyze   — full pipeline: classify + explain + rewrite
  POST /preprocess — NLP pipeline only
  GET  /health    — health check
  GET  /model/info — model metadata
"""

import os
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from preprocessing import preprocess
from model import ToxicityClassifier, train_model, MODEL_PATH, EXTRACTOR_PATH
from explain import Explainer
from rewrite import TextRewriter, semantic_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

clf: ToxicityClassifier | None = None
explainer: Explainer | None = None
rewriter = TextRewriter()


def load_or_train_model():
    """Load saved model or train on demo data if not found."""
    global clf, explainer
    clf = ToxicityClassifier()
    if MODEL_PATH.exists() and EXTRACTOR_PATH.exists():
        try:
            clf.load()
            logger.info("Model loaded from disk.")
        except Exception as e:
            logger.warning(f"Failed to load model ({e}), training on demo data...")
            jigsaw_path = Path(__file__).parent.parent / "data" / "jigsaw_dataset.csv"
            clf = train_model(
                jigsaw_csv=str(jigsaw_path) if jigsaw_path.exists() else None
            )
    else:
        logger.info("No saved model found. Training on demo data...")
        jigsaw_path = Path(__file__).parent.parent / "data" / "jigsaw_dataset.csv"
        clf = train_model(
            jigsaw_csv=str(jigsaw_path) if jigsaw_path.exists() else None
        )
    explainer = Explainer(clf)


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_or_train_model()
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Explainable Toxic Text Normalization API",
    description=(
        "Classical NLP + Interpretable ML pipeline for toxicity detection, "
        "explanation, and rule-based text rewriting. No transformers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, example="You are so stupid!")

class FeatureItem(BaseModel):
    feature: str
    category: str
    value: float
    weight: float
    contribution: float
    human_label: str
    direction: str

class AnalyzeResponse(BaseModel):
    original: str
    toxicity_score: float
    label: str
    cleaned_text: str
    explanation: list[FeatureItem]
    summary: str
    toxic_words: list[str]
    aggressive_patterns_found: list[str]
    sentiment_summary: str
    structural_flags: list[str]
    rewrite_changes: list[str]
    semantic_similarity: float

class PreprocessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class TrainRequest(BaseModel):
    jigsaw_csv: str | None = None
    sample_size: int = 20000
    max_tfidf_features: int = 10000
    C: float = 1.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": clf is not None and clf._trained,
    }


@app.get("/model/info")
def model_info():
    if clf is None or not clf._trained:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": "LogisticRegression",
        "n_features": len(clf.feature_names_) if clf.feature_names_ else "unknown",
        "feature_groups": ["tfidf_ngrams", "linguistic", "toxicity_indicators", "structural"],
        "interpretable": True,
        "uses_transformers": False,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Full pipeline:
    1. Classify toxicity (Logistic Regression)
    2. Generate explanation (feature contributions)
    3. Rewrite text (rule-based)
    4. Compute semantic similarity
    """
    if clf is None or not clf._trained:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # Classify + explain
        exp = explainer.explain(text, top_n=8)
        exp_dict = explainer.to_dict(exp)

        # Rewrite
        rewrite_result = rewriter.rewrite_with_pos(text)
        cleaned = rewrite_result["rewritten"]
        changes = rewrite_result["changes"]

        # Semantic similarity (meaning preservation)
        sim = semantic_similarity(text, cleaned)

        return AnalyzeResponse(
            original=text,
            toxicity_score=exp_dict["toxicity_score"],
            label=exp_dict["label"],
            cleaned_text=cleaned,
            explanation=[FeatureItem(**f) for f in exp_dict["top_features"]],
            summary=exp_dict["summary"],
            toxic_words=exp_dict["toxic_words"],
            aggressive_patterns_found=exp_dict["aggressive_patterns_found"],
            sentiment_summary=exp_dict["sentiment_summary"],
            structural_flags=exp_dict["structural_flags"],
            rewrite_changes=changes,
            semantic_similarity=sim,
        )

    except Exception as e:
        logger.exception("Error during analysis")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess")
def preprocess_text(request: PreprocessRequest):
    """Run only the NLP preprocessing pipeline and return structured output."""
    try:
        result = preprocess(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train_endpoint(request: TrainRequest):
    """
    Retrain the model (useful for updating with new data).
    Provide path to Jigsaw CSV or leave empty for demo data.
    """
    global clf, explainer
    try:
        clf = train_model(
            jigsaw_csv=request.jigsaw_csv,
            sample_size=request.sample_size,
            max_tfidf_features=request.max_tfidf_features,
            C=request.C,
        )
        explainer = Explainer(clf)
        return {"status": "trained", "message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rewrite")
def rewrite_only(request: AnalyzeRequest):
    """Rewrite text without classification (rule-based only)."""
    result = rewriter.rewrite_with_pos(request.text)
    sim = semantic_similarity(result["original"], result["rewritten"])
    result["semantic_similarity"] = sim
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
