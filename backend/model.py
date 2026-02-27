import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ===============================
# Path Configuration
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# ===============================
# Load Datasets (BULLETPROOF)
# ===============================
fake_path = os.path.join(DATA_DIR, "Fake.csv")
true_path = os.path.join(DATA_DIR, "True.csv")

fake = pd.read_csv(
    fake_path,
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

true = pd.read_csv(
    true_path,
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

# ===============================
# Label Assignment
# ===============================
fake["label"] = 0
true["label"] = 1

# ===============================
# Combine & Shuffle
# ===============================
df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ===============================
# Features & Target
# ===============================
X = df["text"]
y = df["label"]

# ===============================
# TF-IDF Vectorization
# ===============================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_vec = vectorizer.fit_transform(X)

# ===============================
# Train Model
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# ===============================
# Save Artifacts
# ===============================
joblib.dump(model, os.path.join(BASE_DIR, "fake_news_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))

print("✅ Fake News Detection model trained and saved successfully")