function checkNews() {
    const text = document.getElementById("newsText").value;

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("resultBox").classList.remove("hidden");
        document.getElementById("prediction").innerText = data.result;
        document.getElementById("confidenceBar").style.width = data.confidence + "%";
        document.getElementById("confidenceText").innerText =
            data.confidence + "% confidence";
    })
    .catch(() => alert("Backend not running"));
}