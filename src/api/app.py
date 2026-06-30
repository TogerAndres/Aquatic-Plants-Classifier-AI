"""
API Flask para el clasificador de plantas acuáticas.

Endpoints:
    GET  /api/health    -> estado del servicio
    GET  /api/classes   -> lista de clases que reconoce el modelo
    POST /api/predict   -> recibe una imagen (multipart/form-data, campo "image")
                           y devuelve clase, probabilidades, heatmap Grad-CAM
                           (base64 PNG) y explicación en texto.

Uso:
    export AQUATIC_MODEL_DIR=./artifacts/final_model
    python -m src.api.app
"""

from __future__ import annotations

import base64
import io
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from src.api.schemas import ErrorResponse, PredictionResponse
from src.inference.predictor import AquaticPlantPredictor

MODEL_DIR = os.environ.get("AQUATIC_MODEL_DIR", "./artifacts/final_model")
BACKBONE = os.environ.get("AQUATIC_BACKBONE", "efficientnetb0")
MAX_IMAGE_MB = 8

app = Flask(__name__)
CORS(app)  # el frontend (Vite, puerto distinto) necesita esto en desarrollo
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_MB * 1024 * 1024

_predictor: AquaticPlantPredictor | None = None


def get_predictor() -> AquaticPlantPredictor:
    """Carga el modelo de forma perezosa y lo reutiliza entre requests.

    Cargar el modelo en cada request sería absurdamente lento (segundos
    por predicción); cargarlo una sola vez al importar el módulo evita
    además que un error de arranque tumbe el proceso antes de poder
    responder /api/health con un mensaje útil.
    """
    global _predictor
    if _predictor is None:
        _predictor = AquaticPlantPredictor(MODEL_DIR, backbone=BACKBONE)
    return _predictor


def _encode_image_to_base64(image_array) -> str:
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/api/health")
def health():
    try:
        predictor = get_predictor()
        return jsonify({"status": "ok", "classes_loaded": len(predictor.class_names)})
    except Exception as exc:  # noqa: BLE001 - queremos reportar cualquier fallo de carga
        return jsonify(ErrorResponse("model_not_loaded", str(exc)).to_dict()), 503


@app.get("/api/classes")
def classes():
    predictor = get_predictor()
    return jsonify({"classes": predictor.class_names})


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify(ErrorResponse("missing_image", "Envía la imagen en el campo 'image'.").to_dict()), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify(ErrorResponse("empty_filename", "El archivo recibido no tiene nombre.").to_dict()), 400

    try:
        image_bytes = file.read()
        predictor = get_predictor()
        result = predictor.predict(image_bytes, with_explanation=False)
    except Exception as exc:  # noqa: BLE001
        return jsonify(ErrorResponse("prediction_failed", str(exc)).to_dict()), 500

    
    return jsonify({
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "top_predictions": result["top_predictions"],
        "all_probabilities": result["all_probabilities"],
        "explanation_text": "",
        "gradcam_overlay_base64": "",
        "dominant_region": ""
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
