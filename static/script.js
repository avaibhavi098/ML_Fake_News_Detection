const SAMPLES = {
    real: "The Department of Labor reported on Tuesday that consumer price inflation slowed to 2.4% annually, matching analysts' expectations according to official data.",
    fake: "SHOCKING BOMBSHELL: Deep state secret documents leaked by anonymous whistleblower! Mainstream media is covering this up, share before it gets deleted!!!",
    uncertain: "City municipal staff scheduled an informal discussion next Thursday concerning potential landscaping upgrades for the district square."
};

function fillSample(type) {
    const textarea = document.getElementById("newsText");
    if (type === "clear") {
        textarea.value = "";
        const resultBox = document.getElementById("resultBox");
        if (resultBox) resultBox.classList.add("hidden");
    } else if (SAMPLES[type]) {
        textarea.value = SAMPLES[type];
        checkNews();
    }
}

function checkNews() {
    const text = document.getElementById("newsText").value;
    const btn = document.getElementById("btnCheck");
    if (btn) btn.innerText = "Analyzing...";

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(res => res.json())
    .then(data => {
        if (btn) btn.innerText = "Check News";
        const resultBox = document.getElementById("resultBox");
        resultBox.classList.remove("hidden");

        const predictionElem = document.getElementById("prediction");
        const barElem = document.getElementById("confidenceBar");
        const textElem = document.getElementById("confidenceText");
        const explainElem = document.getElementById("explainText");

        predictionElem.innerText = data.result;
        barElem.style.width = data.confidence + "%";
        textElem.innerText = data.confidence + "% confidence";

        // Dynamic color and explanation based on classification result
        if (data.result.includes("REAL")) {
            barElem.style.backgroundColor = "#22c55e"; // Green
            if (explainElem) {
                explainElem.innerText = "The model identified balanced journalistic attribution, neutral sentence structures, and lack of sensationalized linguistic markers.";
            }
        } else if (data.result.includes("FAKE")) {
            barElem.style.backgroundColor = "#ef4444"; // Red
            if (explainElem) {
                explainElem.innerText = "The model detected sensational vocabulary, emotional tone, or linguistic patterns commonly associated with misleading headlines.";
            }
        } else if (data.result.includes("UNCERTAIN")) {
            barElem.style.backgroundColor = "#eab308"; // Amber
            if (explainElem) {
                explainElem.innerText = "Confidence is below the certainty threshold (65%). The text contains balanced or neutral phrasing with insufficient distinctive cues to classify reliably.";
            }
        } else {
            barElem.style.backgroundColor = "#64748b";
            if (explainElem) {
                explainElem.innerText = "Please provide an article body or headline to evaluate.";
            }
        }
    })
    .catch((err) => {
        if (btn) btn.innerText = "Check News";
        const resultBox = document.getElementById("resultBox");
        if (resultBox) resultBox.classList.remove("hidden");
        const prediction = document.getElementById("prediction");
        if (prediction) prediction.innerText = "⚠️ Unable to connect to server";
        try {
            alert("Backend not running");
        } catch (_) {}
    });
}
