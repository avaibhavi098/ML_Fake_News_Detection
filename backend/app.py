"""
Production Flask Application for Fake News Detection with Explainable AI (XAI)
Loads serialized model artifacts once on startup.
Applies preprocessing pipeline, calculates feature importance contributions,
and handles confidence thresholds including UNCERTAIN.
"""

import os
import sys
import json
import time
import sqlite3
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from .preprocessing import clean_text
TEMPLATES_DIR = os.path.join(BASE_DIR, "..", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
DB_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "..", "predictions.db"))
MAX_ARTICLE_LENGTH = 50000

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

# In-Memory Rate Limiter
class SimpleRateLimiter:
    def __init__(self, window_seconds=60, max_requests=60):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.clients = {}

    def check(self, ip):
        now = time.time()
        if len(self.clients) > 1000:
            self.clients = {k: v for k, v in self.clients.items() if now - v[0] < self.window_seconds}
        if ip not in self.clients or (now - self.clients[ip][0] > self.window_seconds):
            self.clients[ip] = [now, 1]
            return True, 0
        start_time, count = self.clients[ip]
        if count >= self.max_requests:
            retry_after = int(self.window_seconds - (now - start_time)) + 1
            return False, retry_after
        self.clients[ip][1] += 1
        return True, 0

predict_limiter = SimpleRateLimiter(window_seconds=60, max_requests=60)
ai_limiter = SimpleRateLimiter(window_seconds=60, max_requests=20)

def get_client_ip():
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "127.0.0.1"

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'self' https://ai.studio https://*.google.com https://*.run.app; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    cors_origin = os.environ.get("CORS_ORIGIN")
    if cors_origin:
        response.headers["Access-Control-Allow-Origin"] = cors_origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
        except Exception as e:
            app.logger.warning(f"Could not create database directory {db_dir}: {e}")
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                uncertainty_status INTEGER NOT NULL DEFAULT 0,
                model_version TEXT NOT NULL DEFAULT '2.1.0',
                timestamp TEXT NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (timestamp DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions (prediction);")

init_db()

# Global artifacts loaded ONCE at startup (never retrains on server boot)
MODEL = None
VECTORIZER = None
METADATA = {
    "version": "2.1.0",
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


def explain_instance(cleaned_text, vectorizer, model, top_k=6):
    """
    Computes local feature attribution using TF-IDF weights * Logistic Regression coefficients.
    Returns words supporting REAL, words supporting FAKE, and human-readable explanation.
    """
    disclaimer = "⚠️ Model Evidence vs. Fact-Checking: This analysis evaluates linguistic style, vocabulary choice, and emotional tone. It does NOT cross-reference external databases or confirm factual truth. A factual article can use casual phrasing, and false stories can imitate formal prose."

    if vectorizer is None or model is None:
        return {
            "real_evidence": [],
            "fake_evidence": [],
            "summary": "The prediction is based on heuristic lexical cues as model artifacts are not initialized.",
            "disclaimer": disclaimer
        }

    try:
        vec = vectorizer.transform([cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]  # shape: (n_features,)
        cx = vec.tocoo()

        contributions = []
        for idx, val in zip(cx.col, cx.data):
            weight = coefs[idx] * val
            contributions.append({
                "word": feature_names[idx],
                "score": round(float(weight), 3),
                "support": "real" if weight > 0 else "fake"
            })

        real_evidence = sorted([c for c in contributions if c["score"] > 0], key=lambda x: x["score"], reverse=True)[:top_k]
        fake_evidence = sorted([c for c in contributions if c["score"] < 0], key=lambda x: x["score"])[:top_k]

        # Generate human-readable specific explanation
        if real_evidence and not fake_evidence:
            top_words = ", ".join([f"'{c['word']}'" for c in real_evidence[:3]])
            summary = f"The prediction was influenced by formal, objective textual patterns such as {top_words}, with no prominent sensational markers."
        elif fake_evidence and not real_evidence:
            top_words = ", ".join([f"'{c['word']}'" for c in fake_evidence[:3]])
            summary = f"The prediction was influenced by sensational or emotional language patterns such as {top_words} commonly found in misleading stories."
        elif real_evidence and fake_evidence:
            top_real = ", ".join([f"'{c['word']}'" for c in real_evidence[:2]])
            top_fake = ", ".join([f"'{c['word']}'" for c in fake_evidence[:2]])
            summary = f"The prediction was influenced by competing signals: patterns supporting Real news ({top_real}) versus patterns supporting Fake news ({top_fake})."
        else:
            summary = "The prediction was influenced by general baseline distribution; no high-magnitude distinctive keywords were detected."

        return {
            "real_evidence": real_evidence,
            "fake_evidence": fake_evidence,
            "summary": summary,
            "disclaimer": disclaimer
        }
    except Exception as e:
        app.logger.error(f"Error computing XAI explanation: {e}")
        return {
            "real_evidence": [],
            "fake_evidence": [],
            "summary": "Feature explanation calculation encountered an error.",
            "disclaimer": disclaimer
        }


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


@app.route("/health", methods=["GET"])
def health_check():
    resp = jsonify({"status": "healthy"})
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
    return resp, 200


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Abuse Protection: Rate Limiting
    client_ip = get_client_ip()
    allowed, retry_after = predict_limiter.check(client_ip)
    if not allowed:
        return jsonify({
            "success": False,
            "error": "Rate limit exceeded. Please wait a moment before sending more prediction requests.",
            "retry_after": retry_after
        }), 429

    # 2. Strict Input Validation
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid request body. Expected JSON object."}), 400

    raw_text = data.get("text", "")
    if not isinstance(raw_text, str):
        return jsonify({"success": False, "error": "Field 'text' must be a string."}), 400

    raw_text = raw_text.strip()

    if len(raw_text) > MAX_ARTICLE_LENGTH:
        return jsonify({
            "success": False,
            "error": f"Article text length ({len(raw_text):,} characters) exceeds maximum limit of {MAX_ARTICLE_LENGTH:,} characters."
        }), 400

    if not raw_text:
        return jsonify({
            "result": "⚠️ Enter some news text",
            "prediction": "NONE",
            "confidence": 0,
            "status": "empty_input",
            "uncertainty_status": True,
            "explanation": "Please paste an article headline or body to evaluate.",
            "real_evidence": [],
            "fake_evidence": [],
            "top_real_features": [],
            "top_fake_features": []
        })

    # Clean input text using reusable preprocessing pipeline
    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        return jsonify({
            "result": "⚠️ Enter some news text",
            "prediction": "NONE",
            "confidence": 0,
            "status": "empty_after_cleaning",
            "uncertainty_status": True,
            "explanation": "The text contained only stripped symbols or URLs. Please provide readable text.",
            "real_evidence": [],
            "fake_evidence": [],
            "top_real_features": [],
            "top_fake_features": []
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

        # Compute Explainable AI (XAI) feature importance
        xai = explain_instance(cleaned_text, VECTORIZER, MODEL)
    else:
        result = "UNCERTAIN 🟡"
        confidence = 50
        status = "model_uninitialized"
        xai = explain_instance(cleaned_text, None, None)

    prediction_label = "REAL" if status == "real" else ("FAKE" if status == "fake" else "UNCERTAIN")

    saved_id = None
    if raw_text:
        try:
            with get_db() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO predictions (article_text, prediction, confidence, uncertainty_status, model_version, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        raw_text[:3000],
                        prediction_label,
                        float(confidence),
                        1 if status == "uncertain" else 0,
                        METADATA.get("version", "2.1.0"),
                        datetime.now(timezone.utc).isoformat()
                    )
                )
                conn.commit()
                saved_id = cursor.lastrowid
        except Exception as e:
            app.logger.error(f"Failed to record prediction in DB: {e}")

    return jsonify({
        "result": result,
        "prediction": prediction_label,
        "confidence": confidence,
        "status": status,
        "uncertainty_status": status == "uncertain",
        "model_version": METADATA.get("version", "2.1.0"),
        "explanation": xai["summary"],
        "real_evidence": xai["real_evidence"],
        "fake_evidence": xai["fake_evidence"],
        "top_real_features": xai["real_evidence"],
        "top_fake_features": xai["fake_evidence"],
        "disclaimer": xai["disclaimer"],
        "saved_id": saved_id
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    search = request.args.get("search", "").strip()
    filter_val = request.args.get("filter", "ALL").strip().upper()
    page = max(1, request.args.get("page", 1, type=int))
    limit = min(50, max(1, request.args.get("limit", 10, type=int)))
    offset = (page - 1) * limit

    where_clauses = []
    params = []

    if filter_val in ["REAL", "FAKE", "UNCERTAIN"]:
        where_clauses.append("prediction = ?")
        params.append(filter_val)

    if search:
        where_clauses.append("article_text LIKE ?")
        params.append(f"%{search}%")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with get_db() as conn:
        count_row = conn.execute(f"SELECT COUNT(*) as count FROM predictions {where_sql}", params).fetchone()
        total = count_row["count"] if count_row else 0
        total_pages = max(1, (total + limit - 1) // limit)

        rows = conn.execute(
            f"""
            SELECT id, article_text, prediction, confidence, uncertainty_status, model_version, timestamp
            FROM predictions
            {where_sql}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset]
        ).fetchall()

    items = [
        {
            "id": r["id"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "uncertainty_status": bool(r["uncertainty_status"]),
            "model_version": r["model_version"],
            "timestamp": r["timestamp"],
            "preview": r["article_text"][:130] + "..." if len(r["article_text"]) > 130 else r["article_text"],
            "full_text": r["article_text"]
        }
        for r in rows
    ]

    return jsonify({
        "success": True,
        "items": items,
        "total": total,
        "page": page,
        "totalPages": total_pages,
        "limit": limit
    })


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    with get_db() as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    return jsonify({"success": True, "message": "Prediction history cleared successfully"})


@app.route("/api/history/<int:pred_id>", methods=["DELETE"])
def delete_history_item(pred_id):
    with get_db() as conn:
        conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        conn.commit()
    return jsonify({"success": True, "message": "Prediction record deleted successfully"})


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    with get_db() as conn:
        stats = conn.execute("SELECT COUNT(*) as total, AVG(confidence) as avg_conf FROM predictions").fetchone()
        total = stats["total"] if stats else 0
        avg_conf = round(stats["avg_conf"], 1) if (stats and stats["avg_conf"]) else 0.0

        real_row = conn.execute("SELECT COUNT(*) as c FROM predictions WHERE prediction = 'REAL'").fetchone()
        fake_row = conn.execute("SELECT COUNT(*) as c FROM predictions WHERE prediction = 'FAKE'").fetchone()
        unc_row = conn.execute("SELECT COUNT(*) as c FROM predictions WHERE prediction = 'UNCERTAIN'").fetchone()

        real_count = real_row["c"] if real_row else 0
        fake_count = fake_row["c"] if fake_row else 0
        unc_count = unc_row["c"] if unc_row else 0

        real_pct = round((real_count / total) * 100, 1) if total > 0 else 0.0
        fake_pct = round((fake_count / total) * 100, 1) if total > 0 else 0.0
        unc_pct = round((unc_count / total) * 100, 1) if total > 0 else 0.0

        time_rows = conn.execute(
            """
            SELECT substr(timestamp, 1, 10) as day, prediction, COUNT(*) as c
            FROM predictions
            GROUP BY day, prediction
            ORDER BY day ASC
            """
        ).fetchall()

        recent_rows = conn.execute(
            """
            SELECT id, article_text, prediction, confidence, uncertainty_status, model_version, timestamp
            FROM predictions
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()

    distribution = [
        {"label": "REAL", "count": real_count, "percentage": real_pct, "color": "#10b981"},
        {"label": "FAKE", "count": fake_count, "percentage": fake_pct, "color": "#ef4444"},
        {"label": "UNCERTAIN", "count": unc_count, "percentage": unc_pct, "color": "#f59e0b"}
    ]

    activity_map = {}
    for r in time_rows:
        day = r["day"]
        if day not in activity_map:
            activity_map[day] = {"date": day, "total": 0, "real": 0, "fake": 0, "uncertain": 0}
        c = r["c"]
        activity_map[day]["total"] += c
        pred = r["prediction"]
        if pred == "REAL":
            activity_map[day]["real"] += c
        elif pred == "FAKE":
            activity_map[day]["fake"] += c
        elif pred == "UNCERTAIN":
            activity_map[day]["uncertain"] += c

    activity_over_time = list(activity_map.values())[-10:]

    recent = [
        {
            "id": r["id"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "uncertainty_status": bool(r["uncertainty_status"]),
            "timestamp": r["timestamp"],
            "preview": r["article_text"][:110] + "..." if len(r["article_text"]) > 110 else r["article_text"]
        }
        for r in recent_rows
    ]

    return jsonify({
        "success": True,
        "summary": {
            "total": total,
            "real_count": real_count,
            "fake_count": fake_count,
            "uncertain_count": unc_count,
            "real_pct": real_pct,
            "fake_pct": fake_pct,
            "uncertain_pct": unc_pct,
            "avg_confidence": avg_conf
        },
        "distribution": distribution,
        "activity_over_time": activity_over_time,
        "recent": recent
    })


@app.route("/ai-analysis", methods=["POST"])
def ai_analysis():
    # 1. Abuse Protection: Rate Limiting
    client_ip = get_client_ip()
    allowed, retry_after = ai_limiter.check(client_ip)
    if not allowed:
        return jsonify({
            "error": "Rate limit exceeded for AI assistant. Please wait a moment before trying again.",
            "retry_after": retry_after
        }), 429

    # 2. Strict Input Validation
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({"error": "Invalid request body. Expected a JSON object."}), 400

    raw_text = data.get("text", "")
    if not isinstance(raw_text, str):
        return jsonify({"error": "Field 'text' must be a string."}), 400

    raw_text = raw_text.strip()

    if not raw_text:
        return jsonify({
            "summary": "No article content provided to summarize.",
            "claims": [],
            "language_analysis": [],
            "explanation": "Please provide article text to generate an AI analysis."
        })

    if len(raw_text) > MAX_ARTICLE_LENGTH:
        return jsonify({
            "error": f"Article text length ({len(raw_text):,} characters) exceeds maximum limit of {MAX_ARTICLE_LENGTH:,} characters."
        }), 400

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({
            "summary": "AI analysis is temporarily unavailable (GEMINI_API_KEY not configured). The ML prediction is still available.",
            "claims": [],
            "language_analysis": ["Linguistic analysis is unavailable without an active Gemini connection."],
            "explanation": "AI analysis is temporarily unavailable. The ML prediction is still available."
        })

    try:
        import urllib.request
        import json

        truncated = raw_text[:3500]
        # Security Hardening: Send API key via header instead of exposing in URL query string
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
        prompt = f"""Analyze the following news article text and generate a structured JSON response:
1. summary: A short, clear 2-3 sentence summary of the article.
2. claims: An array of strings extracting the primary factual claims made in the article.
3. language_analysis: An array of strings identifying any sensational, emotional, exaggerated, or manipulative wording (clearly emphasizing this is linguistic style analysis, not factual verification).
4. explanation: A simple, objective explanation describing the narrative framing and perspective of the text.

Do not assert whether external real-world events are true or false. Analyze strictly based on the text provided.

Article Text:
${truncated}"""

        req_body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=req_body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
                "User-Agent": "aistudio-build"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            candidate_text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(candidate_text)
            return jsonify({
                "summary": parsed.get("summary", "Summary unavailable."),
                "claims": parsed.get("claims", []),
                "language_analysis": parsed.get("language_analysis", []),
                "explanation": parsed.get("explanation", "Explanation unavailable.")
            })
    except Exception as e:
        app.logger.error("AI analysis error (safe): Request failed or timed out.")
        return jsonify({
            "summary": "AI analysis is temporarily unavailable. The ML prediction is still available.",
            "claims": [],
            "language_analysis": ["Linguistic analysis could not be completed at this time."],
            "explanation": "AI analysis is temporarily unavailable. The ML prediction is still available."
        })


# Safe Error Handlers (Never expose stack traces)
@app.errorhandler(400)
def bad_request_handler(e):
    return jsonify({"success": False, "error": "Invalid request payload."}), 400


@app.errorhandler(404)
def not_found_handler(e):
    if request.path.startswith("/api/") or request.path in ["/predict", "/ai-analysis"]:
        return jsonify({"success": False, "error": "Endpoint not found."}), 404
    return render_template("index.html"), 404


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"success": False, "error": "Rate limit exceeded."}), 429


@app.errorhandler(500)
def server_error_handler(e):
    return jsonify({"success": False, "error": "An unexpected internal server error occurred."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    debug_val = os.environ.get("FLASK_DEBUG", "false").strip().lower() in ("1", "true")
    app.run(host="0.0.0.0", port=port, debug=debug_val)

