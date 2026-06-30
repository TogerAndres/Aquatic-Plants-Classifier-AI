"""
Entrenamiento con validación cruzada estratificada (K-Fold).

Por qué K-Fold y no un solo split train/val/test:
  Con un dataset relativamente pequeño de 4 especies, un único split
  fijo puede sobreestimar o subestimar la precisión real solo por azar
  en qué imágenes cayeron en validación. K-Fold entrena y evalúa el
  modelo K veces sobre particiones distintas (todas estratificadas, es
  decir, con la misma proporción de cada clase) y promedia el resultado,
  dando una estimación de desempeño mucho más confiable y exponiendo
  si el modelo es inconsistente entre folds (señal de sesgo o de que
  hay clases con pocas imágenes representativas).

Uso:
    python -m src.training.train --backbone efficientnetb0 --folds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.training.config import TrainingConfig
from src.training.dataset import (
    build_manifest,
    class_distribution_report,
    compute_class_weights,
    make_dataset,
    stratified_folds,
)
from src.training.model_factory import (
    build_classifier,
    compile_model,
    get_preprocess_fn,
    unfreeze_for_finetuning,
)
from src.utils.seed import set_global_seed
from src.utils.visualization import plot_training_curves


def parse_args() -> TrainingConfig:
    cfg = TrainingConfig()
    parser = argparse.ArgumentParser(description="Entrena el clasificador de plantas acuáticas.")
    parser.add_argument("--dataset-dir", default=cfg.dataset_dir)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--backbone", default=cfg.backbone, choices=None)
    parser.add_argument("--folds", type=int, default=cfg.n_folds)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--head-epochs", type=int, default=cfg.head_epochs)
    parser.add_argument("--finetune-epochs", type=int, default=cfg.finetune_epochs)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    cfg.dataset_dir = args.dataset_dir
    cfg.output_dir = args.output_dir
    cfg.backbone = args.backbone
    cfg.n_folds = args.folds
    cfg.batch_size = args.batch_size
    cfg.head_epochs = args.head_epochs
    cfg.finetune_epochs = args.finetune_epochs
    cfg.seed = args.seed
    return cfg


def train_one_fold(fold_idx: int, train_ds, val_ds, class_weights, num_classes, cfg: TrainingConfig):
    fold_dir = Path(cfg.output_dir, "folds", f"fold_{fold_idx}")
    fold_dir.mkdir(parents=True, exist_ok=True)

    model = build_classifier(cfg, num_classes)
    compile_model(model, cfg.head_lr, cfg.label_smoothing)

    ckpt_path = str(fold_dir / "best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.early_stopping_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=cfg.reduce_lr_patience, min_lr=1e-7),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    fit_kwargs = {"class_weight": class_weights} if cfg.use_class_weights else {}

    print(f"\n[Fold {fold_idx}] Entrenando cabeza (backbone congelado)...")
    hist_head = model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.head_epochs,
        callbacks=callbacks, verbose=1, **fit_kwargs,
    )

    print(f"[Fold {fold_idx}] Fine-tuning (backbone descongelado)...")
    unfreeze_for_finetuning(model)
    compile_model(model, cfg.finetune_lr, cfg.label_smoothing)
    hist_ft = model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.finetune_epochs,
        callbacks=callbacks, verbose=1, **fit_kwargs,
    )

    plot_training_curves(hist_head, hist_ft, save_path=str(fold_dir / "training_curves.png"))

    val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    print(f"[Fold {fold_idx}] Métricas de validación: {val_metrics}")

    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    return val_metrics


def main():
    cfg = parse_args()
    cfg.ensure_dirs()
    set_global_seed(cfg.seed)

    manifest = build_manifest(cfg.dataset_dir, cfg.class_names or None)
    cfg.class_names = manifest.class_names
    print("Distribución de clases:", class_distribution_report(manifest))

    preprocess_fn = get_preprocess_fn(cfg)
    all_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(stratified_folds(manifest, cfg)):
        print(f"\n{'=' * 60}\nFOLD {fold_idx + 1}/{cfg.n_folds}\n{'=' * 60}")

        train_fp, train_lb = manifest.filepaths[train_idx], manifest.labels[train_idx]
        val_fp, val_lb = manifest.filepaths[val_idx], manifest.labels[val_idx]

        class_weights = compute_class_weights(train_lb, manifest.num_classes) if cfg.use_class_weights else None

        train_ds = make_dataset(train_fp, train_lb, cfg, manifest.num_classes, training=True, preprocess_fn=preprocess_fn)
        val_ds = make_dataset(val_fp, val_lb, cfg, manifest.num_classes, training=False, preprocess_fn=preprocess_fn)

        metrics = train_one_fold(fold_idx, train_ds, val_ds, class_weights, manifest.num_classes, cfg)
        all_fold_metrics.append(metrics)

    # --- Resumen de validación cruzada ---
    summary = {}
    for key in all_fold_metrics[0]:
        values = [m[key] for m in all_fold_metrics]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    print("\n=== RESUMEN VALIDACIÓN CRUZADA ===")
    for key, stats in summary.items():
        print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    with open(Path(cfg.output_dir, "reports", "cross_validation_summary.json"), "w") as f:
        json.dump({"per_fold": all_fold_metrics, "summary": summary, "config": vars(cfg)}, f, indent=2, default=str)

    # --- Entrenamiento final sobre el 100% de los datos ---
    # Una vez validada la arquitectura vía K-Fold, se reentrena una última
    # vez usando todo el dataset (sin separar validación) para producir el
    # modelo que realmente se va a servir en la API. La desviación
    # estándar entre folds, no este modelo final, es la que dice qué tan
    # confiable es la precisión reportada.
    print("\n=== ENTRENAMIENTO FINAL CON TODO EL DATASET ===")
    full_ds = make_dataset(
        manifest.filepaths, manifest.labels, cfg, manifest.num_classes,
        training=True, preprocess_fn=preprocess_fn,
    )
    class_weights = compute_class_weights(manifest.labels, manifest.num_classes) if cfg.use_class_weights else None

    final_model = build_classifier(cfg, manifest.num_classes)
    compile_model(final_model, cfg.head_lr, cfg.label_smoothing)
    fit_kwargs = {"class_weight": class_weights} if cfg.use_class_weights else {}
    final_model.fit(full_ds, epochs=cfg.head_epochs, verbose=1, **fit_kwargs)

    unfreeze_for_finetuning(final_model)
    compile_model(final_model, cfg.finetune_lr, cfg.label_smoothing)
    final_model.fit(full_ds, epochs=cfg.finetune_epochs, verbose=1, **fit_kwargs)

    final_path = Path(cfg.output_dir, "final_model", "model.keras")
    final_model.save(final_path)

    with open(Path(cfg.output_dir, "final_model", "class_names.json"), "w") as f:
        json.dump(manifest.class_names, f, indent=2, ensure_ascii=False)
    with open(Path(cfg.output_dir, "final_model", "model_card.json"), "w") as f:
        json.dump(
            {
                "backbone": cfg.backbone,
                "input_size": cfg.backbone_spec["input_size"],
                "class_names": manifest.class_names,
                "cross_validation_summary": summary,
                "n_folds": cfg.n_folds,
            },
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n✅ Modelo final guardado en: {final_path}")


if __name__ == "__main__":
    main()
