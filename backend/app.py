"""
Production Flask Application for Fake News Detection
Loads serialized model artifacts once on startup.
Applies preprocessing pipeline and handles confidence thresholds including UNCERTAIN.
"""

import os
import json
from flask import Flask, request, jsonify, render_template
import joblib

from preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "..", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

# Global artifacts loaded ONCE at startup (never retrains on server boot)
MODEL = None
VECTORIZER = None
METADATA = {
    "version": "2.0.0",
    "uncertainty_threshold": 65
}

def load_artifacts():
    global MODEL, VECTORIZER, METADATA
    model_path = os.path.join(BASE_DIR, "fake_news_model.pkl")
    vec_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    meta_path = os.path.join(BASE_DIR, "model_metadata.json")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                METADATA = json.load(f)
        except Exception as e:
            app.logger.warning(f"Failed to load model metadata: {e}")

    if os.path.exists(model_path) and os.path.exists(vec_path):
        try:
            MODEL = joblib.load(model_path)
            VECTORIZER = joblib.load(vec_path)
            app.logger.info("Successfully loaded ML model and vectorizer.")
        except Exception as e:
            app.logger.error(f"Error deserializing model artifacts: {e}")
    else:
        app.logger.warning("Model or vectorizer artifacts not found on disk. Run python train.py first.")

# Load on module import
load_artifacts()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Returns version and evaluation metrics."""
    return jsonify({
        "status": "online" if MODEL is not None else "degraded",
        "metadata": METADATA
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    raw_text = data.get("text", "")

    if not raw_text or not raw_text.strip():
        return jsonify({
            "result": "⚠️ Enter some news text",
            "confidence": 0,
            "status": "empty_input"
        })

    # Clean input text using reusable preprocessing pipeline
    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        return jsonify({
            "result": "⚠️ Enter some news text",
            "confidence": 0,
            "status": "empty_after_cleaning"
        })

    # Inference using trained baseline model
    if MODEL is not None and VECTORIZER is not None:
        vector = VECTORIZER.transform([cleaned_text])
        probabilities = MODEL.predict_proba(vector)[0]
        prob_real = probabilities[1]
        prob_fake = probabilities[0]

        max_prob = max(prob_real, prob_fake)
        confidence = round(max_prob * 100)

        uncertainty_cutoff = METADATA.get("uncertainty_threshold", 65)

        # Confidence Thresholding: UNCERTAIN if below threshold
        if confidence < uncertainty_cutoff:
            result = "UNCERTAIN 🟡"
            status = "uncertain"
        elif prob_real > prob_fake:
            result = "REAL NEWS 🟢"
            status = "real"
        else:
            result = "FAKE NEWS 🔴"
            status = "fake"
    else:
        # Fallback heuristic if artifacts not yet built on disk
        result = "UNCERTAIN 🟡"
        confidence = 50
        status = "model_uninitialized"

    return jsonify({
        "result": result,
        "confidence": confidence,
        "status": status,
        "model_version": METADATA.get("version", "2.0.0")
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)
