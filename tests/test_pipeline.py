"""Tests for the scoring app: SQL/Python feature parity, input validation, API behaviour.

Run from the project root with:  pytest -q
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as scoring_app  # noqa: E402

# Same feature SQL as used for training in the notebook (without the label join)
SQL_FEATURES = """
SELECT
    CASE r.type WHEN 'L' THEN 0 WHEN 'M' THEN 1 ELSE 2 END      AS type_code,
    r.air_temperature,
    r.process_temperature,
    r.process_temperature - r.air_temperature                   AS temp_diff,
    r.rotational_speed,
    r.torque,
    r.tool_wear,
    r.torque * r.rotational_speed * 2 * 3.141592653589793 / 60  AS power_w,
    r.torque * r.tool_wear                                      AS strain
FROM readings r
ORDER BY r.udi
"""

NORMAL = {"type": "M", "air_temperature": 298.1, "process_temperature": 308.6,
          "rotational_speed": 1551, "torque": 42.8, "tool_wear": 0}
HIGH_STRAIN = {"type": "L", "air_temperature": 298.0, "process_temperature": 308.5,
               "rotational_speed": 1380, "torque": 66, "tool_wear": 220}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring_app, "LOG_DB", tmp_path / "test_log.db")
    scoring_app.init_db()
    return scoring_app.app.test_client()


def test_sql_and_python_features_match():
    """Training (SQL) and serving (Python) must compute identical features."""
    rng = np.random.default_rng(0)
    rows = [{
        "udi": i + 1,
        "type": str(rng.choice(["L", "M", "H"])),
        "air_temperature": float(rng.uniform(295, 305)),
        "process_temperature": float(rng.uniform(305, 315)),
        "rotational_speed": float(rng.uniform(1100, 2900)),
        "torque": float(rng.uniform(3, 77)),
        "tool_wear": float(rng.uniform(0, 250)),
    } for i in range(200)]

    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE readings (
        udi INTEGER PRIMARY KEY, type TEXT, air_temperature REAL,
        process_temperature REAL, rotational_speed REAL, torque REAL, tool_wear REAL)""")
    pd.DataFrame(rows).to_sql("readings", conn, if_exists="append", index=False)
    sql_features = pd.read_sql_query(SQL_FEATURES, conn)[scoring_app.FEATURES]
    conn.close()

    py_features = pd.concat([scoring_app.build_features(r) for r in rows], ignore_index=True)
    np.testing.assert_allclose(sql_features.to_numpy(float), py_features.to_numpy(float), rtol=1e-9)


@pytest.mark.parametrize("bad_field, bad_value", [
    ("type", "X"),
    ("torque", "abc"),
    ("torque", 500),
    ("rotational_speed", -10),
    ("air_temperature", ""),
])
def test_invalid_input_is_rejected(bad_field, bad_value):
    payload = dict(NORMAL, **{bad_field: bad_value})
    _, errors = scoring_app.parse_reading(payload)
    assert errors


def test_valid_input_is_accepted():
    _, errors = scoring_app.parse_reading(NORMAL)
    assert errors == []


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["features"] == scoring_app.FEATURES


def test_api_rejects_bad_payload(client):
    response = client.post("/api/score", json=dict(NORMAL, torque=500))
    assert response.status_code == 400
    assert "errors" in response.get_json()


def test_api_scores_normal_and_high_strain_readings(client):
    normal = client.post("/api/score", json=NORMAL).get_json()
    strain = client.post("/api/score", json=HIGH_STRAIN).get_json()
    assert normal["anomaly_alert"] is False
    assert normal["failure_probability"] < 0.5
    assert strain["anomaly_alert"] is True
    assert strain["failure_probability"] >= 0.5