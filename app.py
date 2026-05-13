from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from pathlib import Path
import json

# Flask App

app = Flask(__name__)
CORS(app)


# Paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_complaint_classifier_pipeline.pkl"
ENCODER_PATH = BASE_DIR / "models" / "best_label_encoder.pkl"
METRICS_PATH = BASE_DIR / "reports" / "03_best_model_metrics.json"

# Load Model Files

model = joblib.load(MODEL_PATH)

label_encoder = None
if ENCODER_PATH.exists():
    label_encoder = joblib.load(ENCODER_PATH)

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)


# Helper Function

def decode_prediction(prediction):
    """
    Converts encoded numeric prediction into original category name.
    """
    if label_encoder is not None:
        try:
            prediction_int = int(prediction)
            return label_encoder.inverse_transform([prediction_int])[0]
        except Exception:
            return str(prediction)

    return str(prediction)


def get_confidence_scores(text):
    """
    Returns model confidence scores with readable category names.
    """
    if not hasattr(model, "predict_proba"):
        return []

    probabilities = model.predict_proba([text])[0]

    if label_encoder is not None:
        class_names = label_encoder.classes_
    elif hasattr(model, "classes_"):
        class_names = model.classes_
    else:
        class_names = [f"Class {i}" for i in range(len(probabilities))]

    confidence_list = []

    for category, confidence in zip(class_names, probabilities):
        confidence_list.append({
            "category": str(category),
            "confidence": float(confidence),
            "confidence_percent": round(float(confidence) * 100, 2)
        })

    confidence_list = sorted(
        confidence_list,
        key=lambda x: x["confidence"],
        reverse=True
    )

    return confidence_list


# API Routes

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Complaint Classification API is running.",
        "model": "TF-IDF + Logistic Regression",
        "status": "success"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None
    })


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data received."
            }), 400

        complaint_text = data.get("complaint_text", "")

        if complaint_text.strip() == "":
            return jsonify({
                "status": "error",
                "message": "complaint_text cannot be empty."
            }), 400

        prediction = model.predict([complaint_text])[0]
        prediction = decode_prediction(prediction)

        confidence_scores = get_confidence_scores(complaint_text)

        return jsonify({
            "status": "success",
            "input_text": complaint_text,
            "predicted_category": str(prediction),
            "confidence_scores": confidence_scores
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Run Flask App

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )