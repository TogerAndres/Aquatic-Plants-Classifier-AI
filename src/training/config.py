"""
Configuración central del pipeline de entrenamiento.

Toda ruta y todo hiperparámetro vive aquí para que el resto del código
nunca tenga valores "mágicos" embebidos. Esto es lo primero que alguien
debería leer y editar antes de lanzar un entrenamiento.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Backbones soportados. EfficientNetB0 es el default (mejor balance
# precisión/costo para 4 clases con dataset pequeño-mediano), pero se deja
# la puerta abierta a comparar contra otras arquitecturas sin tocar el
# resto del pipeline.
# ---------------------------------------------------------------------------
SUPPORTED_BACKBONES = {
    "efficientnetb0": {
        "module": "tensorflow.keras.applications.efficientnet",
        "class_name": "EfficientNetB0",
        "preprocess": "preprocess_input",
        "input_size": (224, 224),
        "last_conv_layer": "top_conv",
    },
    "efficientnetb1": {
        "module": "tensorflow.keras.applications.efficientnet",
        "class_name": "EfficientNetB1",
        "preprocess": "preprocess_input",
        "input_size": (240, 240),
        "last_conv_layer": "top_conv",
    },
    "resnet50v2": {
        "module": "tensorflow.keras.applications.resnet_v2",
        "class_name": "ResNet50V2",
        "preprocess": "preprocess_input",
        "input_size": (224, 224),
        "last_conv_layer": "post_relu",
    },
    "mobilenetv3large": {
        "module": "tensorflow.keras.applications.mobilenet_v3",
        "class_name": "MobileNetV3Large",
        "preprocess": "preprocess_input",
        "input_size": (224, 224),
        "last_conv_layer": "expanded_conv_14/Add" ,
    },
}


@dataclass
class TrainingConfig:
    # --- Datos ---
    dataset_dir: str = os.environ.get(
        "AQUATIC_DATASET_DIR", "./data/Augmented Images"
    )
    output_dir: str = "./artifacts"
    class_names: list[str] = field(
        default_factory=lambda: [
            "Common Duckweeds (Lemna minor)",
            "Common Water Hyacinth (Eichornia crassipes)",
            "Heartleaf False Pickerelweed (Monochoria korsakowii)",
            "Water Lettuce (Pistia stratiotes)",
        ]
    )

    # --- Arquitectura ---
    backbone: str = "efficientnetb0"
    dense_units: tuple[int, ...] = (512, 256)
    dropout_rates: tuple[float, ...] = (0.4, 0.3)
    l2_reg: float = 1e-3

    # --- Validación cruzada ---
    n_folds: int = 5
    stratified: bool = True
    seed: int = 42

    # --- Entrenamiento ---
    batch_size: int = 16
    head_epochs: int = 15          # Solo cabeza (backbone congelado)
    finetune_epochs: int = 25      # Backbone descongelado
    head_lr: float = 1e-3
    finetune_lr: float = 1e-5
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    label_smoothing: float = 0.05  # Ayuda con clases visualmente similares

    # --- Aumentación de datos ---
    # El dataset ya viene aumentado (carpeta "Augmented Images"), así que
    # aquí solo se aplican aumentos ligeros adicionales en tiempo real
    # para que cada época vea variaciones distintas y se reduzca el
    # sobreajuste a las copias aumentadas fijas.
    rotation_range: int = 20
    zoom_range: float = 0.15
    horizontal_flip: bool = True
    brightness_range: tuple[float, float] = (0.8, 1.2)
    shear_range: float = 0.1
    use_class_weights: bool = True  # Mitiga sesgo por clases desbalanceadas

    @property
    def backbone_spec(self) -> dict:
        try:
            return SUPPORTED_BACKBONES[self.backbone.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Backbone '{self.backbone}' no soportado. "
                f"Opciones: {list(SUPPORTED_BACKBONES)}"
            ) from exc

    @property
    def input_shape(self) -> tuple[int, int, int]:
        h, w = self.backbone_spec["input_size"]
        return (h, w, 3)

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "folds").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "final_model").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "reports").mkdir(parents=True, exist_ok=True)
