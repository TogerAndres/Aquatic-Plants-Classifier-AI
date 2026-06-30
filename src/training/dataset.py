"""
Construye un manifiesto (filepath, label) a partir de la carpeta de
imágenes y arma pipelines tf.data reproducibles para cada fold.

Trabajar a partir de un manifiesto en vez de `flow_from_directory` es lo
que permite hacer K-Fold real: necesitamos la lista completa de archivos
con su clase para poder partirla con StratifiedKFold y garantizar que
cada fold mantenga la misma proporción de clases (evita que un fold quede
con pocas o ninguna imagen de una clase rara, que es una fuente directa
de sesgo en datasets pequeños).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from src.training.config import TrainingConfig


@dataclass
class Manifest:
    filepaths: np.ndarray
    labels: np.ndarray          # índices enteros
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def build_manifest(dataset_dir: str, class_names: list[str] | None = None) -> Manifest:
    """Escanea `dataset_dir/<clase>/*.jpg` y construye el manifiesto.

    Si `class_names` es None, las clases se infieren de los nombres de
    subcarpeta (ordenados alfabéticamente) para que el dataset_dir sea la
    única fuente de verdad.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"No existe el directorio del dataset: '{dataset_dir}'. "
            "Actualiza AQUATIC_DATASET_DIR o TrainingConfig.dataset_dir."
        )

    found_classes = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    if not found_classes:
        raise ValueError(f"No se encontraron subcarpetas de clases en '{dataset_dir}'.")

    class_names = class_names or found_classes
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

    filepaths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            raise ValueError(
                f"Clase '{cls}' declarada en config pero no existe en '{dataset_dir}'."
            )
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(valid_ext):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(idx)

    if not filepaths:
        raise ValueError("El dataset está vacío: no se encontraron imágenes válidas.")

    return Manifest(
        filepaths=np.array(filepaths),
        labels=np.array(labels, dtype=np.int64),
        class_names=class_names,
    )


def stratified_folds(manifest: Manifest, cfg: TrainingConfig):
    """Genera (train_idx, val_idx) para cada fold, balanceado por clase."""
    skf = StratifiedKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
    )
    yield from skf.split(manifest.filepaths, manifest.labels)


def class_distribution_report(manifest: Manifest) -> dict[str, int]:
    """Conteo de imágenes por clase, útil para detectar desbalance."""
    counts = np.bincount(manifest.labels, minlength=manifest.num_classes)
    return {cls: int(c) for cls, c in zip(manifest.class_names, counts)}


def compute_class_weights(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    """Pesos inversamente proporcionales a la frecuencia de cada clase.

    Se usan en `model.fit(..., class_weight=...)` para que el modelo no
    favorezca a la clase mayoritaria: es la mitigación de sesgo más simple
    y efectiva cuando no se puede recolectar más datos de la clase rara.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def _load_and_preprocess(filepath: tf.Tensor, label: tf.Tensor, input_size: tuple[int, int]):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, input_size)
    image = tf.cast(image, tf.float32)
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor, cfg: TrainingConfig):
    if cfg.horizontal_flip:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(
        image, max_delta=(cfg.brightness_range[1] - 1.0)
    )
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    angle_deg = tf.random.uniform([], -cfg.rotation_range, cfg.rotation_range)
    # Rotación ligera vía tf.image (evita dependencia de tf-addons)
    image = tf.image.rot90(
        image, k=tf.cast(tf.round(angle_deg / 90.0), tf.int32) % 4
    ) if cfg.rotation_range >= 90 else image
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def make_dataset(
    filepaths: np.ndarray,
    labels: np.ndarray,
    cfg: TrainingConfig,
    num_classes: int,
    training: bool,
    preprocess_fn,
) -> tf.data.Dataset:
    """Pipeline tf.data: lectura -> resize -> (aumentación) -> preprocess -> batch."""
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=cfg.seed)

    input_size = cfg.backbone_spec["input_size"]
    ds = ds.map(
        lambda fp, lb: _load_and_preprocess(fp, lb, input_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        ds = ds.map(lambda img, lb: _augment(img, lb, cfg), num_parallel_calls=tf.data.AUTOTUNE)

    def _finalize(img, lb):
        img = preprocess_fn(img)
        lb = tf.one_hot(lb, depth=num_classes)
        return img, lb

    ds = ds.map(_finalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
