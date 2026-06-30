"""
Evaluación detallada del modelo final: reporte de clasificación,
matriz de confusión y distribución de confianza por clase.

Esto es independiente del K-Fold de `train.py`: aquí se evalúa el
modelo final ya entrenado con todo el dataset, idealmente contra un
conjunto de imágenes nuevas que el modelo nunca vio (un holdout externo
o fotos propias), no contra datos de entrenamiento.

Uso:
    python -m src.training.evaluate --model-path artifacts/final_model/model.keras \
        --test-dir ./data/test_holdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from src.training.config import TrainingConfig
from src.training.dataset import build_manifest, make_dataset
from src.training.model_factory import get_preprocess_fn
from src.utils.visualization import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--backbone", default="efficientnetb0")
    parser.add_argument("--output-dir", default="./artifacts/reports")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cfg = TrainingConfig(backbone=args.backbone, dataset_dir=args.test_dir)
    manifest = build_manifest(args.test_dir)
    preprocess_fn = get_preprocess_fn(cfg)

    test_ds = make_dataset(
        manifest.filepaths, manifest.labels, cfg, manifest.num_classes,
        training=False, preprocess_fn=preprocess_fn,
    )

    model = tf.keras.models.load_model(args.model_path)
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = manifest.labels
    confidences = np.max(probs, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=manifest.class_names, output_dict=True
    )
    print(classification_report(y_true, y_pred, target_names=manifest.class_names))

    plot_confusion_matrix(
        y_true, y_pred, manifest.class_names,
        save_path=str(Path(args.output_dir) / "confusion_matrix.png"),
    )

    results_df = pd.DataFrame({
        "filepath": manifest.filepaths,
        "true_class": [manifest.class_names[i] for i in y_true],
        "predicted_class": [manifest.class_names[i] for i in y_pred],
        "confidence": confidences,
        "correct": y_true == y_pred,
    })
    results_df.to_csv(Path(args.output_dir) / "predictions.csv", index=False)

    with open(Path(args.output_dir) / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Reporte guardado en {args.output_dir}")


if __name__ == "__main__":
    main()
