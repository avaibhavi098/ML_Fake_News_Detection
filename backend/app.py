from flask import Flask, request, jsonify, render_template
import joblib, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "templates"),
    static_folder=os.path.join(BASE_DIR, "..", "static")
)

model = joblib.load(os.path.join(BASE_DIR, "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"result": "⚠️ Enter some news text", "confidence": 0})

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence = round(max(model.predict_proba(vector)[0]) * 100)

    result = "REAL NEWS 🟢" if prediction == 1 else "FAKE NEWS 🔴"

    return jsonify({
        "result": result,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)