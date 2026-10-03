"""
app.py  —  Flask REST API for House Rent Prediction
----------------------------------------------------
Endpoints:
  GET  /health           → service health check
  GET  /options          → dropdown values for all categorical features
  POST /predict          → predict rent from JSON body
  GET  /feature_importance → feature importance from the trained model
"""

import os
import re
import joblib
import numpy as np
from flask import Flask, request, jsonify

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "rent_model.pkl")
ENC_PATH   = os.path.join(BASE_DIR, "model", "encoders.pkl")

# ── Load artefacts ───────────────────────────────────────────────────────────
model    = joblib.load(MODEL_PATH)
encoders = joblib.load(ENC_PATH)

FEATURE_COLS = encoders["feature_cols"]
CAT_COLS     = encoders["cat_cols"]

app = Flask(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_floor(floor_str: str):
    floor_str = str(floor_str).lower().strip()
    if "ground" in floor_str:
        current = 0
    else:
        m = re.match(r"(\d+)", floor_str)
        current = int(m.group(1)) if m else 0
    m_total = re.search(r"out of\s+(\d+)", floor_str)
    total = int(m_total.group(1)) if m_total else current
    return current, total


def encode_input(data: dict) -> np.ndarray:
    """Transform raw user input dict into a model-ready numpy array."""
    floor_current, floor_total = parse_floor(data.get("floor", "1 out of 2"))

    encoded_cats = {}
    for col in CAT_COLS:
        le    = encoders[col]
        value = data.get(col, le.classes_[0])
        if value not in le.classes_:
            value = le.classes_[0]
        encoded_cats[col + "_enc"] = int(le.transform([value])[0])

    row = {
        "BHK":          int(data.get("BHK", 2)),
        "Size":         float(data.get("Size", 800)),
        "Bathroom":     int(data.get("Bathroom", 1)),
        "Floor_Current": floor_current,
        "Floor_Total":   floor_total,
        **encoded_cats,
    }
    return np.array([[row[f] for f in FEATURE_COLS]])


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "GradientBoostingRegressor"})


@app.route("/options", methods=["GET"])
def options():
    """Return valid dropdown choices for all categorical features."""
    return jsonify(encoders["unique_values"])


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON body:
    {
      "BHK": 2,
      "Size": 900,
      "Bathroom": 2,
      "floor": "2 out of 5",
      "Area Type": "Super Area",
      "City": "Mumbai",
      "Furnishing Status": "Semi-Furnished",
      "Tenant Preferred": "Bachelors/Family",
      "Point of Contact": "Contact Owner"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        X = encode_input(data)
        prediction = float(model.predict(X)[0])
        prediction = max(0, round(prediction, 2))

        return jsonify({
            "predicted_rent": prediction,
            "currency": "INR",
            "formatted": f"Rs. {prediction:,.0f}"
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    importance = model.feature_importances_.tolist()
    return jsonify({
        "features": FEATURE_COLS,
        "importance": importance
    })


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🏠 House Rent Prediction API running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
