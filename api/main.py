"""
FastAPI backend for Gairaigo Map.

Endpoints:
    GET  /health      - liveness check
    GET  /languages   - returns the 3 classifiable languages
    POST /predict     - classifies a katakana word
    POST /emotion     - detects emotion from plain text, returns music list + loanwords

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
from transformers import pipeline


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / "models"

MODEL_PATH = MODELS_DIR / "model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
ENCODER_PATH = MODELS_DIR / "encoder.joblib"

KATAKANA_RE = re.compile(r"^[\u30A0-\u30FF\u30FC\u30FB\u30FE\u30FD]+$")


def is_katakana(text: str) -> bool:
    return bool(KATAKANA_RE.match(text.strip()))


# ── Language metadata (SVM classifier) ────────────────────────────────────────
LANGUAGE_META = {
    "English": {"iso2": "GB", "country": "United Kingdom", "color": "#4a90d9"},
    "French":  {"iso2": "FR", "country": "France",          "color": "#e85d5d"},
    "German":  {"iso2": "DE", "country": "Germany",         "color": "#f0a500"},
}

# ── Emotion → Music playlist (multiple songs per emotion for variety) ──────────
# All video IDs verified via Wikipedia / official sources
EMOTION_MUSIC: dict[str, list[dict]] = {
    "joy": [
        {"title": "ATARASHII GAKKO! - Que Sera Sera", "video_id": "0S1-b9xGQac"},
        {"title": "Wonderland x Showtime - Kyoufuu All Back", "video_id": "nq_x3D0_lgw"},
        {"title": "Creepy Nuts - Bling-Bang-Bang-Born", "video_id": "mLW35YMzELE"},
        {"title": "Wonderland x Showtime - Taiyoukei Disco", "video_id": "oA6aCY4bMg4"},
        {"title": "Gen Hoshino - Koi", "video_id": "jhOVibLEDhA"},
        {"title": "Yumi Arai - Rouge no Dengon", "video_id": "MH-P4mXvDPE"},
    ],
    "sadness": [
        {"title": "ZONE - Kimi ga Kureta Mono", "video_id": "Of36Qh7WLSQ"},
        {"title": "YOSHIKI - Red Swan", "video_id": "r1XE8ON8fos"},
        {"title": "Galileo Galilei - Aoi Shiori", "video_id": "T3bxbVGWy5k"},
        {"title": "DAOKO x Kenshi Yonezu - Fireworks", "video_id": "-tKVN2mAKRI"},
        {"title": "Yorushika - Say It", "video_id": "F64yFFnZfkI"},
        {"title": "Kenshi Yonezu - Lemon", "video_id": "SX_ViT4Ra7k"},
        {"title": "Yoh Kamiyama - Irokousui", "video_id": "kQYLHjgUh_g"},
    ],
    "anger": [
        {"title": "Ado - Usseewa", "video_id": "Qp3b-RXtz4w"},
        {"title": "Neru - Lost One's Weeping", "video_id": "U1aS62Juz70"},
        {"title": "Minami - Crying for Rain", "video_id": "0YF8vecQWYs"},
        {"title": "Eve - Dramaturgy", "video_id": "jJzw1h5CR-I"},
        {"title": "Kenshi Yonezu - Kick Back", "video_id": "M2cckDmNLMI"},
    ],
    "fear": [
        {"title": "Nightcord at 25:00 x KAITO - Heat Abnormal", "video_id": "ToqKNyZi2NQ"},
        {"title": "Yuzu - Hyori Ittai", "video_id": "eKoD2CRr_KA"},
        {"title": "Nightcord at 25:00 - Bug", "video_id": "2Ii7UBMxWVw"},
        {"title": "RADWIMPS - Nandemonaiya", "video_id": "n89SKAymNfA"},
        {"title": "sakanaction - Arukuaround", "video_id": "cADu9rtlZGQ"},
    ],
    "surprise": [
        {"title": "Ado - New Genesis", "video_id": "1FliVTcX8bQ"},
        {"title": "RADWIMPS - Grand Escape", "video_id": "epQGR34yiTY"},
        {"title": "Ado - Buriki no Dance", "video_id": "iL7uoLCbJoc"},
        {"title": "YOASOBI - Idol", "video_id": "ZRtdQ81jPUQ"},
    ],
    "disgust": [
        {"title": "Ado - Readymade", "video_id": "jg09lNupc1s"},
        {"title": "Nightcord at 25:00 - Bocca della Verità", "video_id": "ZjNUJUgyoOw"},
        {"title": "Eve - Literary Nonsense", "video_id": "OskXF3s0UT8"},
    ],
    "neutral": [
        {"title": "ATARASHII GAKKO - Dounimo Tomaranai", "video_id": "59bnq4wlGx8"},
        {"title": "Mitchie M - Tokugawa Cup Noodle Kinshirei", "video_id": "jPXAgWkqbo4"},
        {"title": "Homecomings - Cakes", "video_id": "u1A53wFN9A0"},
        {"title": "Hanae - Kamisama Hajimemashita", "video_id": "gZaelu4lieE"},
    ],
}

# ── Emotion → curated loanwords ────────────────────────────────────────────────
EMOTION_LOANWORDS: dict[str, list[dict]] = {
    "joy": [
        {"katakana": "カーニバル",    "meaning": "carnival",                          "language": "English",    "iso2": "GB"},
        {"katakana": "フェスティバル", "meaning": "festival",                          "language": "English",    "iso2": "GB"},
        {"katakana": "ダンス",        "meaning": "dance",                             "language": "English",    "iso2": "GB"},
        {"katakana": "ショーロ",      "meaning": "choro; chorinho; style of Brazilian popular music", "language": "Portuguese", "iso2": "PT"},
        {"katakana": "カステラ",      "meaning": "castella (type of sponge cake)",    "language": "Portuguese", "iso2": "PT"},
        {"katakana": "バレエ",        "meaning": "ballet",                            "language": "French",     "iso2": "FR"},
        {"katakana": "シャンソン",    "meaning": "chanson; French song",              "language": "French",     "iso2": "FR"},
        {"katakana": "フェット",      "meaning": "fête; festival; celebration",       "language": "French",     "iso2": "FR"},
    ],
    "sadness": [
        {"katakana": "ノスタルジー",  "meaning": "nostalgia",                         "language": "French",     "iso2": "FR"},
        {"katakana": "メランコリー",  "meaning": "melancholy",                        "language": "French",     "iso2": "FR"},
        {"katakana": "アデュー",      "meaning": "adieu; goodbye",                    "language": "French",     "iso2": "FR"},
        {"katakana": "ミンネ",        "meaning": "love of a knight for a courtly lady (upon which he is unable to act)", "language": "German", "iso2": "DE"},
        {"katakana": "フロイライン",  "meaning": "miss (German title for an unmarried woman)", "language": "German", "iso2": "DE"},
        {"katakana": "カッパ",        "meaning": "raincoat",                          "language": "Portuguese", "iso2": "PT"},
        {"katakana": "ロンリー",      "meaning": "lonely",                            "language": "English",    "iso2": "GB"},
        {"katakana": "ブルース",      "meaning": "blues (music genre)",               "language": "English",    "iso2": "GB"},
    ],
    "anger": [
        {"katakana": "ネリチャギ",    "meaning": "axe kick; ax kick",                 "language": "Korean",     "iso2": "KR"},
        {"katakana": "サンダ",        "meaning": "sanda; sanshou; Chinese boxing; Chinese kickboxing", "language": "Chinese", "iso2": "CN"},
        {"katakana": "テロル",        "meaning": "terror; terrorism",                 "language": "German",     "iso2": "DE"},
        {"katakana": "ストライキ",    "meaning": "strike (labor action)",             "language": "English",    "iso2": "GB"},
        {"katakana": "プロテスト",    "meaning": "protest",                           "language": "English",    "iso2": "GB"},
        {"katakana": "レジスタンス",  "meaning": "resistance (movement)",             "language": "French",     "iso2": "FR"},
        {"katakana": "バトル",        "meaning": "battle",                            "language": "English",    "iso2": "GB"},
        {"katakana": "パワー",        "meaning": "power",                             "language": "English",    "iso2": "GB"},
    ],
    "fear": [
        {"katakana": "ノワール",      "meaning": "black; dark",                       "language": "French",     "iso2": "FR"},
        {"katakana": "エトランゼ",    "meaning": "stranger; outsider; foreigner",     "language": "French",     "iso2": "FR"},
        {"katakana": "テロル",        "meaning": "terror; terrorism",                 "language": "German",     "iso2": "DE"},
        {"katakana": "デマゴギー",    "meaning": "false rumor; false alarm; misinformation", "language": "German", "iso2": "DE"},
        {"katakana": "ゴースト",      "meaning": "ghost",                             "language": "English",    "iso2": "GB"},
        {"katakana": "ホラー",        "meaning": "horror",                            "language": "English",    "iso2": "GB"},
        {"katakana": "パニック",      "meaning": "panic",                             "language": "English",    "iso2": "GB"},
        {"katakana": "ミステリー",    "meaning": "mystery",                           "language": "English",    "iso2": "GB"},
    ],
    "surprise": [
        {"katakana": "ゲリラライブ",  "meaning": "surprise concert",                  "language": "English",    "iso2": "GB"},
        {"katakana": "スライハンド",  "meaning": "sleight of hand (e.g. in magic tricks)", "language": "English", "iso2": "GB"},
        {"katakana": "マジック",      "meaning": "magic",                             "language": "English",    "iso2": "GB"},
        {"katakana": "イリュージョン", "meaning": "illusion",                         "language": "English",    "iso2": "GB"},
        {"katakana": "サーカス",      "meaning": "circus",                            "language": "English",    "iso2": "GB"},
        {"katakana": "スペクタクル",  "meaning": "spectacle",                         "language": "French",     "iso2": "FR"},
        {"katakana": "ブリュット",    "meaning": "brut; dry sparkling wine",          "language": "French",     "iso2": "FR"},
        {"katakana": "サプライズ",    "meaning": "surprise",                          "language": "English",    "iso2": "GB"},
    ],
    "disgust": [
        {"katakana": "チョウドウフ",  "meaning": "stinky tofu; fermented tofu",       "language": "Chinese",    "iso2": "CN"},
        {"katakana": "シックハウス",  "meaning": "sick building; building which causes people to feel unwell", "language": "English", "iso2": "GB"},
        {"katakana": "トキシック",    "meaning": "toxic",                             "language": "English",    "iso2": "GB"},
        {"katakana": "ダストシュート", "meaning": "garbage chute; trash chute",       "language": "English",    "iso2": "GB"},
        {"katakana": "ポイズン",      "meaning": "poison",                            "language": "English",    "iso2": "GB"},
        {"katakana": "スキャンダル",  "meaning": "scandal",                           "language": "English",    "iso2": "GB"},
        {"katakana": "ネガティブ",    "meaning": "negative",                          "language": "English",    "iso2": "GB"},
        {"katakana": "ゴミ",          "meaning": "rubbish; trash; garbage",           "language": "English",    "iso2": "GB"},
    ],
    "neutral": [
        {"katakana": "アルバイター",  "meaning": "part-time worker; part-timer",      "language": "German",     "iso2": "DE"},
        {"katakana": "ピンイン",      "meaning": "Pinyin (Chinese romanization system)", "language": "Chinese", "iso2": "CN"},
        {"katakana": "スケジュール",  "meaning": "schedule",                          "language": "English",    "iso2": "GB"},
        {"katakana": "システム",      "meaning": "system",                            "language": "English",    "iso2": "GB"},
        {"katakana": "ドキュメント",  "meaning": "document",                          "language": "English",    "iso2": "GB"},
        {"katakana": "ネットワーク",  "meaning": "network",                           "language": "English",    "iso2": "GB"},
        {"katakana": "マネジメント",  "meaning": "management",                        "language": "English",    "iso2": "GB"},
        {"katakana": "スタンダード",  "meaning": "standard",                          "language": "English",    "iso2": "GB"},
    ],
}

# ── Startup ────────────────────────────────────────────────────────────────────
artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in (MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH):
        if not path.exists():
            raise RuntimeError(f"Model artifact not found: {path}")
    artifacts["model"] = joblib.load(MODEL_PATH)
    artifacts["vectorizer"] = joblib.load(VECTORIZER_PATH)
    artifacts["encoder"] = joblib.load(ENCODER_PATH)
    artifacts["emotion"] = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
    )
    print("✓ Model artifacts loaded")
    print("✓ Emotion classifier loaded")
    yield
    artifacts.clear()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Gairaigo Map API", version="2.0.0", lifespan=lifespan)

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


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    word: str

    @field_validator("word")
    @classmethod
    def must_be_katakana(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Word must not be empty.")
        if not is_katakana(v):
            raise ValueError("Input must be katakana (e.g. コーヒー).")
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


class EmotionRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Text must not be empty.")
        return v


class MusicEntry(BaseModel):
    title: str
    video_id: str


class LoanwordResult(BaseModel):
    katakana: str
    meaning: str
    language: str
    iso2: str


class EmotionResponse(BaseModel):
    text: str
    emotion: str
    music_list: list[MusicEntry]   # full playlist — frontend cycles through these
    loanwords: list[LoanwordResult]


# ── Helpers ────────────────────────────────────────────────────────────────────
def softmax(scores: np.ndarray) -> np.ndarray:
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def classify(word: str) -> PredictResponse:
    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    encoder = artifacts["encoder"]

    X = vectorizer.transform([word])
    decision_scores = model.decision_function(X)[0]
    confidences = softmax(decision_scores)
    classes = encoder.classes_

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
    all_scores.sort(key=lambda r: r.confidence, reverse=True)
    return PredictResponse(word=word, prediction=all_scores[0], all_scores=all_scores)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "model_loaded": bool(artifacts)}


@app.get("/languages", tags=["Meta"])
def get_languages():
    return {lang: meta for lang, meta in LANGUAGE_META.items()}


@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
def predict(body: PredictRequest):
    try:
        return classify(body.word)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotion", response_model=EmotionResponse, tags=["Emotion"])
def detect_emotion(body: EmotionRequest):
    """
    Detect emotion from plain English text.
    Returns the detected emotion, a playlist of matching Japanese songs, and related loanwords.
    The frontend can cycle through music_list to let users skip to the next song.
    """
    try:
        result = artifacts["emotion"](body.text)
        label: str = result[0][0]["label"].lower()

        if label not in EMOTION_MUSIC:
            label = "neutral"

        music_list = [MusicEntry(**m) for m in EMOTION_MUSIC[label]]
        loanwords = [LoanwordResult(**w) for w in EMOTION_LOANWORDS[label]]

        return EmotionResponse(
            text=body.text,
            emotion=label,
            music_list=music_list,
            loanwords=loanwords,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
