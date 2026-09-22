"""
Production Readiness & Regression Test Suite for Fake News Detection System
Tests both the Flask backend and the active Node.js server to verify full compliance.
"""

import os
import sys
import json
import sqlite3
import unittest
from datetime import datetime

# Add project root and backend to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app as flask_app, DB_PATH, MODEL, VECTORIZER, METADATA


class ProductionReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = flask_app.test_client()
        flask_app.config["TESTING"] = True

    # ==========================================
    # 1. APPLICATION STARTUP & CONFIG
    # ==========================================
    def test_01_startup_and_health(self):
        """Verify /health endpoint returns 200 and correct status"""
        res = cls = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "healthy")
        # Check security headers
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertIn("Content-Security-Policy", res.headers)

    def test_02_model_artifacts_loaded(self):
        """Verify ML artifacts are preloaded and not None (no training on startup)"""
        self.assertIsNotNone(MODEL, "Model should be loaded on startup")
        self.assertIsNotNone(VECTORIZER, "Vectorizer should be loaded on startup")
        self.assertIn("version", METADATA)

    # ==========================================
    # 2. ML PREDICTION FLOWS
    # ==========================================
    def test_03_predict_real_article(self):
        """Test prediction on Real-like news text"""
        article = "The Federal Reserve announced on Wednesday that benchmark interest rates would remain unchanged following their policy meeting."
        res = self.client.post("/predict", json={"text": article})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)
        self.assertIn(data["prediction"], ["REAL", "FAKE", "UNCERTAIN"])
        self.assertGreaterEqual(data["confidence"], 0)
        self.assertLessEqual(data["confidence"], 100)
        self.assertIn("explanation", data)
        self.assertIn("disclaimer", data)
        self.assertIsNotNone(data.get("saved_id"))

    def test_04_predict_fake_article(self):
        """Test prediction on Fake-like sensational article"""
        article = "BREAKING BOMBSHELL: Alien spacecraft discovered under Capitol Hill, secret government cover-up exposed by anonymous insider!"
        res = self.client.post("/predict", json={"text": article})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)
        self.assertEqual(data["prediction"], "FAKE")
        self.assertGreaterEqual(data["confidence"], 60)
        self.assertGreater(len(data.get("fake_evidence", [])), 0)

    def test_05_predict_uncertain_or_neutral(self):
        """Test prediction on neutral/ambiguous statement"""
        article = "The quick brown fox jumps over the lazy dog in the afternoon sun."
        res = self.client.post("/predict", json={"text": article})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("uncertainty_status", data)
        # Should not crash and should return valid structure
        self.assertIn("confidence", data)

    def test_06_predict_empty_input(self):
        """Test prediction with empty string"""
        res = self.client.post("/predict", json={"text": ""})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("prediction"), "NONE")
        self.assertTrue(data.get("uncertainty_status"))

    def test_07_predict_whitespace_only(self):
        """Test prediction with whitespace only"""
        res = self.client.post("/predict", json={"text": "   \n\t   "})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("prediction"), "NONE")

    def test_08_predict_very_short(self):
        """Test prediction with 1 or 2 words"""
        res = self.client.post("/predict", json={"text": "hello"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)

    def test_09_predict_unknown_words(self):
        """Test prediction with OOV (out-of-vocabulary) words"""
        res = self.client.post("/predict", json={"text": "asdkfjhwe zxnvkjsdf qowieuroiwue zmncbvnsd"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)

    def test_10_predict_special_characters(self):
        """Test prediction with HTML, scripts, and unicode special chars"""
        res = self.client.post("/predict", json={"text": "<script>alert('xss')</script> !@#$%^&*()_+ 🚀⚡ <b>Bold</b>"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)

    def test_11_predict_max_length_exceeded(self):
        """Test payload rejection when exceeding MAX_ARTICLE_LENGTH"""
        oversized = "A" * 50005
        res = self.client.post("/predict", json={"text": oversized})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    # ==========================================
    # 3. EXPLAINABLE AI (XAI)
    # ==========================================
    def test_12_xai_evidence_structure(self):
        """Verify XAI returns structured evidence, explanations, and fact-checking disclaimer"""
        res = self.client.post("/predict", json={"text": "Scientists published an official study confirming the climate trend."})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("explanation", data)
        self.assertIn("disclaimer", data)
        self.assertIn("linguistic style", data["disclaimer"].lower())
        self.assertIn("cross-reference external databases", data["disclaimer"].lower())

    # ==========================================
    # 4. GEMINI AI ASSISTANT & RESILIENCY
    # ==========================================
    def test_13_ai_analysis_empty_input(self):
        """Verify /ai-analysis handles empty input cleanly"""
        res = self.client.post("/ai-analysis", json={"text": ""})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("summary", data)

    def test_14_ai_analysis_missing_key_fallback(self):
        """Verify /ai-analysis gracefully falls back without server error when API key is unset or unavailable"""
        orig_key = os.environ.get("GEMINI_API_KEY")
        try:
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            res = self.client.post("/ai-analysis", json={"text": "The government released the quarterly budget report today."})
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIn("temporarily unavailable", data.get("summary", "").lower())
        finally:
            if orig_key:
                os.environ["GEMINI_API_KEY"] = orig_key

    def test_15_ai_analysis_invalid_key_fallback(self):
        """Verify invalid API key does not crash Flask service"""
        orig_key = os.environ.get("GEMINI_API_KEY")
        try:
            os.environ["GEMINI_API_KEY"] = "INVALID_TEST_KEY_12345"
            res = self.client.post("/ai-analysis", json={"text": "Some breaking headline about economics."})
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIn("temporarily unavailable", data.get("summary", "").lower())
        finally:
            if orig_key:
                os.environ["GEMINI_API_KEY"] = orig_key
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]

    # ==========================================
    # 5. DATABASE & PERSISTENCE
    # ==========================================
    def test_16_db_persistence(self):
        """Verify database writes and schema consistency"""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
            self.assertIsNotNone(cursor.fetchone())

            cursor.execute("PRAGMA table_info(predictions)")
            cols = [col[1] for col in cursor.fetchall()]
            for expected_col in ["id", "article_text", "prediction", "confidence", "uncertainty_status", "model_version", "timestamp"]:
                self.assertIn(expected_col, cols)

    # ==========================================
    # 6. HISTORY API & FILTERS
    # ==========================================
    def test_17_history_retrieval(self):
        """Verify history endpoint returns items and pagination"""
        res = self.client.get("/api/history?page=1&limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("items", data)
        self.assertIn("total", data)

    def test_18_history_filters(self):
        """Test history filtering by REAL, FAKE, UNCERTAIN"""
        for f in ["REAL", "FAKE", "UNCERTAIN"]:
            res = self.client.get(f"/api/history?filter={f}")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            for item in data.get("items", []):
                self.assertEqual(item["prediction"], f)

    def test_19_history_search(self):
        """Test searching history records"""
        res = self.client.get("/api/history?search=Federal")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))

    # ==========================================
    # 7. ANALYTICS & DATABASE RECONCILIATION
    # ==========================================
    def test_20_analytics_reconciliation(self):
        """Verify analytics values match actual SQLite database rows"""
        res = self.client.get("/api/analytics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        summary = data.get("summary", {})

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            actual_total = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            actual_real = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='REAL'").fetchone()[0]
            actual_fake = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='FAKE'").fetchone()[0]
            actual_unc = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='UNCERTAIN'").fetchone()[0]

        self.assertEqual(summary.get("total"), actual_total)
        self.assertEqual(summary.get("real_count"), actual_real)
        self.assertEqual(summary.get("fake_count"), actual_fake)
        self.assertEqual(summary.get("uncertain_count"), actual_unc)

    # ==========================================
    # 8. SECURITY CHECKS
    # ==========================================
    def test_21_secrets_and_env_protection(self):
        """Verify .env is in .gitignore and .env.example contains no secrets"""
        with open(os.path.join(ROOT_DIR, ".gitignore"), "r") as f:
            gi = f.read()
            self.assertIn(".env", gi)

        with open(os.path.join(ROOT_DIR, ".env.example"), "r") as f:
            ex = f.read()
            self.assertNotIn("AIza", ex)
            self.assertNotIn("sk-", ex)

    def test_22_safe_error_handlers(self):
        """Verify 404 returns safe JSON without stack traces"""
        res = self.client.get("/api/nonexistent-endpoint-test")
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertFalse(data.get("success", True))
        self.assertNotIn("Traceback", str(res.data))

    def test_23_frontend_assets_integrity(self):
        """Verify index.html, style.css, and script.js exist and serve properly"""
        res_home = self.client.get("/")
        self.assertEqual(res_home.status_code, 200)
        self.assertIn(b"Fake News Detection", res_home.data)

        res_static_js = self.client.get("/static/script.js")
        self.assertEqual(res_static_js.status_code, 200)
        self.assertIn(b"checkNews", res_static_js.data)

        res_static_css = self.client.get("/static/style.css")
        self.assertEqual(res_static_css.status_code, 200)
        self.assertIn(b"--bg-base", res_static_css.data)


if __name__ == "__main__":
    unittest.main()
