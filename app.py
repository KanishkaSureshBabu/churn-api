from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = ["login_frequency", "feature_adoption", "support_tickets",
            "days_since_active", "contract_months"]

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Churn Prediction API"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    values = np.array([[data[f] for f in FEATURES]])
    prediction = int(model.predict(values)[0])
    probability = round(float(model.predict_proba(values)[0][1]), 4)
    return jsonify({
        "churn_prediction": prediction,
        "churn_probability": probability,
        "risk_level": "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)