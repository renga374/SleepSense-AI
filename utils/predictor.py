"""
Sleep Quality Prediction Engine (FIXED VERSION)
"""
import os
import joblib
import json
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# PATH SETUP (FIXED)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "sleep_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

# ─────────────────────────────────────────────
# LOAD ARTIFACTS SAFELY
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Meta file not found: {META_PATH}")

_model = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)

with open(META_PATH, "r") as f:
    _meta = json.load(f)

FEATURE_COLS = _meta["feature_cols"]
QUALITY_LABELS = {int(k): v for k, v in _meta["quality_labels"].items()}

# ─────────────────────────────────────────────
# ENCODING MAPS
# ─────────────────────────────────────────────
CAFFEINE_MAP = {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3}
MOOD_MAP = {'Happy': 3, 'Neutral': 2, 'Sad': 1, 'Anxious': 0}
INTERRUPT_MAP = {'No': 0, 'Yes': 1}

# ─────────────────────────────────────────────
# SCORE ENGINE
# ─────────────────────────────────────────────
def compute_sleep_score(features: dict) -> dict:
    score = 50.0
    pos, neg = [], []

    dur = float(features.get("sleep_duration", 0))

    if 7 <= dur <= 9:
        score += 15; pos.append("Healthy sleep duration")
    elif 6 <= dur < 7 or 9 < dur <= 10:
        score += 5; pos.append("Acceptable sleep duration")
    else:
        score -= 10; neg.append("Suboptimal sleep duration")

    stress = float(features.get("stress_level", 5))
    score -= stress * 2.5

    if stress <= 3:
        pos.append("Low stress level")
    elif stress >= 7:
        neg.append("High stress level")

    ex = float(features.get("exercise_duration", 0))
    if ex >= 30:
        score += 12; pos.append("Regular exercise")
    elif ex >= 15:
        score += 6; pos.append("Moderate activity")
    else:
        score -= 5; neg.append("Insufficient exercise")

    st = float(features.get("screen_time_before_bed", 0))
    if st <= 30:
        score += 10; pos.append("Low screen time")
    elif st <= 60:
        score += 3
    elif st <= 120:
        score -= 5; neg.append("High screen time")
    else:
        score -= 12; neg.append("Excessive screen time")

    caf = features.get("caffeine_intake", "None")
    if caf == "High":
        score -= 12; neg.append("High caffeine intake")
    elif caf == "Moderate":
        score -= 5
    else:
        score += 3; pos.append("Low caffeine intake")

    mood = features.get("mood_before_sleep", "Neutral")
    if mood in ["Sad", "Anxious"]:
        score -= 8; neg.append("Negative mood before sleep")
    elif mood == "Happy":
        score += 6; pos.append("Positive mood")

    score = int(np.clip(score, 0, 100))

    return {
        "score": score,
        "positive_factors": pos[:4],
        "negative_factors": neg[:4]
    }


# ─────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────
def predict_sleep_quality(features: dict) -> dict:

    vec = [
        float(features["sleep_duration"]),
        float(features["bedtime_hour"]),
        float(features["wake_hour"]),
        float(CAFFEINE_MAP.get(features["caffeine_intake"], 0)),
        float(features["exercise_duration"]),
        float(features["screen_time_before_bed"]),
        float(features["stress_level"]),
        float(MOOD_MAP.get(features["mood_before_sleep"], 2)),
        float(INTERRUPT_MAP.get(features["sleep_interruptions"], 0)),
    ]

    df = pd.DataFrame([vec], columns=FEATURE_COLS)
    scaled = _scaler.transform(df)

    pred = _model.predict(scaled)[0]
    proba = _model.predict_proba(scaled)[0]

    quality = QUALITY_LABELS[int(pred)]
    confidence = int(max(proba) * 100)

    score_data = compute_sleep_score(features)

    return {
        "quality": quality,
        "confidence": confidence,
        "sleep_score": score_data["score"],
        "positive_factors": score_data["positive_factors"],
        "negative_factors": score_data["negative_factors"],
        "class_probs": {
            "Poor": int(proba[0] * 100),
            "Average": int(proba[1] * 100),
            "Good": int(proba[2] * 100),
        }
    }


def get_model_meta():
    return _meta