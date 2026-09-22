# Fake News Detection System

An AI-powered news credibility and fake news detection web service built with **Python Flask**, **Scikit-Learn**, and **Explainable AI (XAI)**. The system analyzes article text using TF-IDF feature extraction and Logistic Regression to detect linguistic manipulation, sensationalism, and credibility markers. It includes local feature attribution, uncertainty detection, an analytics dashboard, prediction history, and optional supplementary analysis powered by Google Gemini.

---

## 1. Production Architecture Overview

- **Primary Production Server**: **Python Flask** served via **Gunicorn WSGI**.
- **Model Execution**: Serialized ML artifacts (`fake_news_model.pkl`, `vectorizer.pkl`, and `model_metadata.json`) loaded once into memory during application startup. No training occurs at runtime or startup.
- **Database**: Embedded **SQLite** (`predictions.db`), automatically created and managed by the application.
- **Node.js Note**: `server.js` is an optional development/container utility. The Python Flask + Gunicorn stack is completely self-contained and **does NOT require Node.js** in production.

---

## 2. Requirements & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- pip

### Install Dependencies
From the repository root:

```bash
pip install -r requirements.txt
```

The dependencies specified in `requirements.txt` are:
- `flask`: Web application framework
- `flask-cors`: Cross-Origin Resource Sharing handling
- `scikit-learn`: ML model and vectorizer runtime
- `pandas`: Data handling pipeline
- `numpy`: Numerical computations
- `joblib`: Model artifact serialization/deserialization
- `gunicorn`: Production WSGI HTTP server

---

## 3. Running Locally

### Option A: Direct Python (Development)
```bash
python3 backend/app.py
```
By default, this binds to `0.0.0.0:3000` (or the port specified in `PORT`) with `FLASK_DEBUG=false`.

### Option B: Local Production Simulation with Gunicorn
To test in the exact production configuration:
```bash
PORT=5055 FLASK_DEBUG=false gunicorn --bind 0.0.0.0:5055 --workers 1 backend.app:app
```
Then visit `http://localhost:5055` in your browser.

> **Note on Workers**: Always use a single worker (`--workers 1`) on free-tier platforms or when using SQLite to avoid database write concurrency conflicts and stay well within memory limits.

---

## 4. Production Start Command

From the repository root:

```bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 backend.app:app
```

---

## 5. Environment Variables

Configure these variables in your hosting dashboard or local `.env` file (see `.env.example`):

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `PORT` | Auto-provided | `3000` | Port on which the HTTP server listens. Provided automatically by cloud platforms (Render, Cloud Run, Heroku). |
| `FLASK_DEBUG` | Optional | `false` | Set to `false` or `0` for production to suppress debugger pins and stack traces. |
| `GEMINI_API_KEY` | Optional | *None* | **Secret**: Google Gemini API key for supplementary AI contextual breakdowns. If omitted, the app operates normally with full ML + XAI capabilities. |
| `CORS_ORIGIN` | Optional | *None* | Restricts allowed cross-origin request domains. Leave empty to restrict to same-origin. |
| `DATABASE_PATH` | Optional | `predictions.db` | Absolute or relative path to the SQLite database file. Useful when mounting persistent cloud disks. |

> **Security Reminder**: Only `GEMINI_API_KEY` is a secret. Never commit `.env` or API keys to source control.

---

## 6. How to Deploy to Render

### Method 1: Using Blueprint (`render.yaml`)
1. Push this repository to GitHub or GitLab.
2. In the Render Dashboard, click **New +** → **Blueprint**.
3. Select your repository. Render will automatically detect `render.yaml` and configure:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 backend.app:app`
   - **Health Check Path**: `/health`
4. Under Environment Variables, add `GEMINI_API_KEY` (optional) as a secret.
5. Click **Apply**.

### Method 2: Manual Web Service Setup
1. In Render Dashboard, click **New +** → **Web Service**.
2. Connect your repository.
3. Configure settings:
   - **Name**: `fake-news-detection`
   - **Region**: Select closest to your users
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 backend.app:app`
   - **Plan**: `Free`
4. In **Advanced Settings**:
   - **Health Check Path**: `/health`
   - **Environment Variables**:
     - `FLASK_DEBUG`: `false`
     - `GEMINI_API_KEY`: *(Your Google AI Studio Gemini API Key, optional)*
5. Click **Deploy Web Service**.

---

## 7. Health Check Endpoint

- **Endpoint**: `GET /health`
- **Response Status**: `200 OK`
- **Response Body**:
  ```json
  {
    "status": "healthy"
  }
  ```
- Render uses this endpoint during zero-downtime deploys and service monitoring.

---

## 8. Gemini Configuration & Graceful Fallback

The application optionally queries the Gemini API (`gemini-2.5-flash`) at `/ai-analysis` for stylistic breakdowns, credibility red flags, and verification advice.

- **With Key**: If `GEMINI_API_KEY` is set, requests receive enhanced AI-generated insights.
- **Without Key / Key Failure**: If `GEMINI_API_KEY` is not provided or fails (e.g. rate limits or invalid key), the endpoint gracefully returns a safe, pre-structured fallback notice. **Core ML predictions, feature attribution (XAI), prediction history, and analytics remain 100% operational.**

---

## 9. Key Machine Learning & XAI Limitations

1. **Stylistic Classification vs. External Fact-Checking**:
   The model evaluates vocabulary choices, emotional sensationalism, and stylistic patterns learned from benchmark news corpora. It does **not** query external live web databases or cross-reference real-time news sources. A factual article written casually might trigger uncertainty, while fabricated news disguised in formal journalistic tone may score high credibility.
2. **Uncertainty Classification**:
   Articles with ambiguous feature balances (confidence between 40% and 65%) or very sparse content are explicitly flagged as `UNCERTAIN` to prevent false certainty.
3. **Publisher Bias Mitigation**:
   Standard news agency signatures (e.g., `(Reuters) -`, `AP`) are stripped by `backend/preprocessing.py` before inference to prevent the classifier from memorizing source names rather than analyzing content.
4. **SQLite Persistence**:
   SQLite is stored locally on the container filesystem. On free-tier ephemeral cloud instances (like Render Free), the database will reset when the instance spins down after inactivity unless a persistent disk is attached.
