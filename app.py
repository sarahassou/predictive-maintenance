"""Predictive-maintenance scoring app.

Scores one machine reading with (a) an Isolation Forest trained on normal data
only and (b) a supervised Random Forest baseline, and logs each request to SQLite.
Features are computed exactly as in the SQL used for training.
"""
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
LOG_DB = BASE_DIR / "scoring_log.db"

bundle = joblib.load(BASE_DIR / "model_bundle.joblib")
IFOREST = bundle["iforest"]
RF = bundle["rf"]
FEATURES = bundle["features"]
IF_THRESHOLD = bundle["if_threshold"]
FALSE_ALARM_RATE = bundle["false_alarm_rate"]

try:
    METRICS = json.loads((BASE_DIR / "metrics.json").read_text())
except FileNotFoundError:
    METRICS = []

TYPE_CODES = {"L": 0, "M": 1, "H": 2}

# Loose physical sanity bounds, same as the validation step in the notebook
BOUNDS = {
    "air_temperature": ("Air temperature (K)", 280, 320),
    "process_temperature": ("Process temperature (K)", 290, 330),
    "rotational_speed": ("Rotational speed (rpm)", 500, 4000),
    "torque": ("Torque (Nm)", 0, 100),
    "tool_wear": ("Tool wear (min)", 0, 300),
}

app = Flask(__name__)


def init_db():
    with sqlite3.connect(LOG_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scored_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scored_at TEXT NOT NULL,
                type TEXT NOT NULL,
                air_temperature REAL, process_temperature REAL,
                rotational_speed REAL, torque REAL, tool_wear REAL,
                anomaly_score REAL, anomaly_alert INTEGER,
                failure_probability REAL
            )
        """)


def parse_reading(payload):
    """Validate raw input; return (reading, errors)."""
    errors = []
    reading = {}

    machine_type = str(payload.get("type", "")).strip().upper()
    if machine_type not in TYPE_CODES:
        errors.append("Type must be L, M or H.")
    reading["type"] = machine_type

    for key, (label, lo, hi) in BOUNDS.items():
        raw = payload.get(key, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if not (lo <= value <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
        reading[key] = value
    return reading, errors


def build_features(reading):
    """Same derived features as the SQL used for training."""
    row = {
        "type_code": TYPE_CODES[reading["type"]],
        "air_temperature": reading["air_temperature"],
        "process_temperature": reading["process_temperature"],
        "temp_diff": reading["process_temperature"] - reading["air_temperature"],
        "rotational_speed": reading["rotational_speed"],
        "torque": reading["torque"],
        "tool_wear": reading["tool_wear"],
        "power_w": reading["torque"] * reading["rotational_speed"] * 2 * math.pi / 60,
        "strain": reading["torque"] * reading["tool_wear"],
    }
    return pd.DataFrame([row])[FEATURES]


def score_reading(reading):
    frame = build_features(reading)
    anomaly_score = float(-IFOREST.score_samples(frame)[0])
    anomaly_alert = anomaly_score >= IF_THRESHOLD
    failure_probability = float(RF.predict_proba(frame)[0, 1])
    return {
        "anomaly_score": round(anomaly_score, 4),
        "anomaly_threshold": round(IF_THRESHOLD, 4),
        "anomaly_alert": bool(anomaly_alert),
        "failure_probability": round(failure_probability, 4),
        "supervised_alert": failure_probability >= 0.5,
        "derived": {k: round(float(frame.iloc[0][k]), 2)
                    for k in ("temp_diff", "power_w", "strain")},
    }


def log_result(reading, result):
    with sqlite3.connect(LOG_DB) as conn:
        conn.execute(
            """INSERT INTO scored_readings
               (scored_at, type, air_temperature, process_temperature,
                rotational_speed, torque, tool_wear,
                anomaly_score, anomaly_alert, failure_probability)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             reading["type"], reading["air_temperature"],
             reading["process_temperature"], reading["rotational_speed"],
             reading["torque"], reading["tool_wear"],
             result["anomaly_score"], int(result["anomaly_alert"]),
             result["failure_probability"]),
        )


def recent_history(limit=10):
    with sqlite3.connect(LOG_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scored_readings ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def render_page(form=None, result=None, errors=None):
    return render_template(
        "index.html",
        form=form or {}, result=result, errors=errors or [],
        history=recent_history(), metrics=METRICS,
        false_alarm_pct=int(FALSE_ALARM_RATE * 100),
    )


@app.get("/")
def index():
    return render_page()


@app.post("/predict")
def predict():
    form = request.form.to_dict()
    reading, errors = parse_reading(form)
    if errors:
        return render_page(form=form, errors=errors), 400
    result = score_reading(reading)
    log_result(reading, result)
    return render_page(form=form, result=result)


@app.post("/api/score")
def api_score():
    payload = request.get_json(silent=True) or {}
    reading, errors = parse_reading(payload)
    if errors:
        return jsonify({"errors": errors}), 400
    result = score_reading(reading)
    log_result(reading, result)
    return jsonify(result)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "features": FEATURES})


init_db()

if __name__ == "__main__":
    app.run(debug=True)