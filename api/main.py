"""
FastAPI backend that loads the saved model artifacts
and serves predictions for the Discover Mode of the Gairaigo Map.

Endpoints:
    GET  /health           — liveness check
    GET  /languages        — returns the 3 classifiable languages
    POST /predict          — classifies a katakana word

Usage:
    uvicorn main:app --reload --port 8000
"""

import re
import numpy as np
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator


# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / "models"

MODEL_PATH = MODELS_DIR / "model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
ENCODER_PATH = MODELS_DIR / "encoder.joblib"

# Katakana validation
KATAKANA_RE = re.compile(r"^[\u30A0-\u30FF\u30FC\u30FB\u30FE\u30FD]+$")


def is_katakana(text: str) -> bool:
    return bool(KATAKANA_RE.match(text.strip()))


# Language metadata for the three classifiable languages
# (mirrors what the frontend needs to highlight the map)
LANGUAGE_META = {
    "English": {"iso2": "GB", "country": "United Kingdom", "color": "#4a90d9"},
    "French": {"iso2": "FR", "country": "France", "color": "#e85d5d"},
    "German": {"iso2": "DE", "country": "Germany", "color": "#f0a500"},
}


# Lifespan, load model artifacts once on startup
artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in (MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH):
        if not path.exists():
            raise RuntimeError(
                f"Model artifact not found: {path}\n"
                "Run `python -m scripts.train` from your kataklassifer project first."
            )
    artifacts["model"] = joblib.load(MODEL_PATH)
    artifacts["vectorizer"] = joblib.load(VECTORIZER_PATH)
    artifacts["encoder"] = joblib.load(ENCODER_PATH)
    print("✓ Model artifacts loaded")
    yield
    artifacts.clear()


# App
app = FastAPI(
    title="Gairaigo Map API",
    description="Classifies Japanese katakana loanwords into English, French, or German.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://kotabi.vercel.app",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Schemas
class PredictRequest(BaseModel):
    word: str

    @field_validator("word")
    @classmethod
    def must_be_katakana(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Word must not be empty.")
        if not is_katakana(v):
            raise ValueError(
                "Input must be a katakana string (e.g. コーヒー). "
                "Hiragana, kanji, or romaji are not supported."
            )
        return v


class LanguageResult(BaseModel):
    language: str
    country: str
    iso2: str
    confidence: float
    color: str


class PredictResponse(BaseModel):
    word: str
    prediction: LanguageResult
    all_scores: list[LanguageResult]


# Helpers
def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert raw SVM decision scores to a probability-like distribution."""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def classify(word: str) -> PredictResponse:
    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    encoder = artifacts["encoder"]

    X = vectorizer.transform([word])

    # decision_function returns shape (1, n_classes) for multi-class LinearSVC
    decision_scores = model.decision_function(X)[0]  # shape: (3,)
    confidences = softmax(decision_scores)  # normalized to sum=1
    _ = int(np.argmax(confidences))
    classes = encoder.classes_  # e.g. ["English", "French", "German"]

    all_scores = [
        LanguageResult(
            language=classes[i],
            country=LANGUAGE_META[classes[i]]["country"],
            iso2=LANGUAGE_META[classes[i]]["iso2"],
            confidence=round(float(confidences[i]), 4),
            color=LANGUAGE_META[classes[i]]["color"],
        )
        for i in range(len(classes))
    ]

    # Sort descending by confidence for the frontend
    all_scores.sort(key=lambda r: r.confidence, reverse=True)

    return PredictResponse(
        word=word,
        prediction=all_scores[0],
        all_scores=all_scores,
    )


# Routes
@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "model_loaded": bool(artifacts)}


@app.get("/languages", tags=["Meta"])
def get_languages():
    """Returns metadata for the 3 classifiable donor languages."""
    return {lang: meta for lang, meta in LANGUAGE_META.items()}


@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
def predict(body: PredictRequest):
    """
    Classify a single katakana loanword.

    - **word**: A katakana string, e.g. `コーヒー`, `アルバイト`, `テレビ`
    - Returns the predicted donor language with a softmax confidence score,
      plus all 3 languages ranked by confidence.
    """
    try:
        return classify(body.word)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
