"""
Predictor de alto nivel: carga el modelo final una sola vez y expone un
único método `predict(image)` que devuelve clase, probabilidades,
heatmap Grad-CAM y una explicación en texto. Este es el objeto que la
API y cualquier script de inferencia deberían usar; nadie por fuera de
este archivo debería tocar Keras directamente.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference.explainability import GradCAM
from src.training.config import TrainingConfig
from src.training.model_factory import get_preprocess_fn


class AquaticPlantPredictor:
    def __init__(self, model_dir: str, backbone: str = "efficientnetb0"):
        import tensorflow as tf  # import perezoso: evita cargar TF solo por importar el módulo

        model_dir = Path(model_dir)
        model_card_path = model_dir / "model_card.json"
        if model_card_path.exists():
            with open(model_card_path) as f:
                card = json.load(f)
            backbone = card.get("backbone", backbone)
            self.class_names: list[str] = card["class_names"]
        else:
            with open(model_dir / "class_names.json") as f:
                self.class_names = json.load(f)

        self.cfg = TrainingConfig(backbone=backbone)
        self.model = tf.keras.models.load_model(model_dir / "model.keras")
        self.preprocess_fn = get_preprocess_fn(self.cfg)
        self.gradcam = GradCAM(self.model, self.cfg)
        self.input_size = self.cfg.backbone_spec["input_size"]

    def _load_image(self, image: "str | bytes | Image.Image") -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            import io
            return Image.open(io.BytesIO(image)).convert("RGB")
        return Image.open(image).convert("RGB")

    def predict(self, image, top_k: int = 4, with_explanation: bool = True) -> dict:
        pil_image = self._load_image(image)
        resized = pil_image.resize(self.input_size[::-1])  # PIL usa (w, h)
        original_rgb = np.array(resized)

        preprocessed = self.preprocess_fn(original_rgb.astype(np.float32))
        batch = np.expand_dims(preprocessed, axis=0)

        probs = self.model.predict(batch, verbose=0)[0]
        order = np.argsort(probs)[::-1][:top_k]

        top_predictions = [
            {"class": self.class_names[i], "confidence": float(probs[i])}
            for i in order
        ]
        predicted_index = int(order[0])

        result = {
            "predicted_class": self.class_names[predicted_index],
            "confidence": float(probs[predicted_index]),
            "top_predictions": top_predictions,
            "all_probabilities": {
                self.class_names[i]: float(p) for i, p in enumerate(probs)
            },
        }

        if with_explanation:
            heatmap, _ = self.gradcam.compute_heatmap(preprocessed, class_index=predicted_index)
            overlay = self.gradcam.overlay_heatmap(original_rgb, heatmap)
            focus = self.gradcam.region_focus_summary(heatmap)

            result["explainability"] = {
                "gradcam_overlay": overlay,           # np.ndarray RGB uint8, listo para encode
                "dominant_region": focus["dominant_region"],
                "region_scores": focus["region_scores"],
                "explanation_text": self._build_explanation_text(
                    result["predicted_class"], result["confidence"], focus["dominant_region"]
                ),
            }

        return result

    @staticmethod
    def _build_explanation_text(predicted_class: str, confidence: float, dominant_region: str) -> str:
        region_es = {
            "centro": "el centro de la imagen",
            "borde_superior": "la parte superior de la imagen",
            "borde_inferior": "la parte inferior de la imagen",
            "borde_izquierdo": "el lado izquierdo de la imagen",
            "borde_derecho": "el lado derecho de la imagen",
        }[dominant_region]

        if confidence >= 0.85:
            certeza = "con alta confianza"
        elif confidence >= 0.6:
            certeza = "con confianza moderada"
        else:
            certeza = "con baja confianza; se recomienda verificar manualmente"

        return (
            f"El modelo clasificó la imagen como '{predicted_class}' {certeza} "
            f"({confidence * 100:.1f}%). La decisión se basó principalmente en "
            f"{region_es}, según el mapa de activación (Grad-CAM)."
        )
