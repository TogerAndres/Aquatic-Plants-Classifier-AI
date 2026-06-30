"""Estructuras de respuesta de la API, centralizadas para que el contrato
con el frontend quede documentado en un solo lugar."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PredictionResponse:
    predicted_class: str
    confidence: float
    top_predictions: list
    all_probabilities: dict
    explanation_text: str
    gradcam_overlay_base64: str
    dominant_region: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ErrorResponse:
    error: str
    detail: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
