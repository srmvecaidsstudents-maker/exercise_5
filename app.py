from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Titanic Survival Predictor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0a0a1a; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
    .card { background: #12122a; border: 1px solid #2a2a5a; border-radius: 16px; padding: 40px; width: 420px; box-shadow: 0 0 40px rgba(80,80,255,0.1); }
    h1 { font-size: 22px; color: #a0a8ff; margin-bottom: 6px; }
    p.sub { color: #666; font-size: 13px; margin-bottom: 28px; }
    label { display: block; font-size: 13px; color: #999; margin-bottom: 6px; margin-top: 16px; }
    input, select { width: 100%; background: #1a1a35; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; color: #e0e0e0; font-size: 14px; }
    input:focus, select:focus { outline: none; border-color: #5050ff; }
    button { margin-top: 24px; width: 100%; background: #4040cc; border: none; border-radius: 8px; padding: 12px; color: white; font-size: 15px; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #5050ff; }
    .result { margin-top: 20px; border-radius: 10px; padding: 16px; text-align: center; font-size: 15px; font-weight: 600; display: none; }
    .survived { background: rgba(0,200,100,0.15); border: 1px solid #00c864; color: #00e870; }
    .died { background: rgba(255,60,60,0.15); border: 1px solid #ff3c3c; color: #ff6060; }
    .prob { font-size: 12px; font-weight: normal; margin-top: 4px; color: #999; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🚢 Titanic Survival Predictor</h1>
    <p class="sub">Gaussian Naive Bayes · 77.6% accuracy</p>
    <label>Passenger Class</label>
    <select id="pclass">
      <option value="1">1st Class</option>
      <option value="2">2nd Class</option>
      <option value="3" selected>3rd Class</option>
    </select>
    <label>Age</label>
    <input type="number" id="age" value="28" min="1" max="100">
    <label>Fare (£)</label>
    <input type="number" id="fare" value="14.50" step="0.01">
    <label>Sex</label>
    <select id="sex">
      <option value="1">Female</option>
      <option value="0">Male</option>
    </select>
    <button onclick="predict()">Predict Survival</button>
    <div class="result" id="result">
      <div id="verdict"></div>
      <div class="prob" id="prob"></div>
    </div>
  </div>
  <script>
    async function predict() {
      const data = {
        Pclass: parseInt(document.getElementById('pclass').value),
        Age: parseFloat(document.getElementById('age').value),
        Fare: parseFloat(document.getElementById('fare').value),
        female: parseInt(document.getElementById('sex').value)
      };
      const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data) });
      const json = await res.json();
      const box = document.getElementById('result');
      box.style.display = 'block';
      box.className = 'result ' + (json.survived ? 'survived' : 'died');
      document.getElementById('verdict').textContent = json.survived ? '✓ Likely Survived' : '✗ Likely Did Not Survive';
      document.getElementById('prob').textContent = `Survival probability: ${json.survival_probability}%`;
    }
  </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    import pandas as pd
    features = pd.DataFrame([[data["Pclass"], data["Age"], data["Fare"], data["female"]]], columns=["Pclass","Age","Fare","female"])
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    return jsonify({
        "survived": prediction,
        "survival_probability": round(float(proba[1]) * 100, 1),
        "death_probability": round(float(proba[0]) * 100, 1)
    })

if __name__ == "__main__":
    app.run(debug=True)
