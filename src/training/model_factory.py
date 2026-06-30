"""
Construye el modelo de clasificación a partir de cualquier backbone
soportado en `config.SUPPORTED_BACKBONES`. Mantener esto separado del
script de entrenamiento permite reutilizar exactamente la misma
arquitectura en inferencia y en notebooks de experimentación.
"""

from __future__ import annotations

import importlib

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.regularizers import l2

from src.training.config import TrainingConfig


def load_backbone(cfg: TrainingConfig, trainable: bool = False) -> tf.keras.Model:
    spec = cfg.backbone_spec
    module = importlib.import_module(spec["module"])
    backbone_cls = getattr(module, spec["class_name"])
    base_model = backbone_cls(
        weights="imagenet",
        include_top=False,
        input_shape=cfg.input_shape,
    )

    base_model._name = "backbone"
    base_model.trainable = trainable
    return base_model


def get_preprocess_fn(cfg: TrainingConfig):
    spec = cfg.backbone_spec
    module = importlib.import_module(spec["module"])
    return getattr(module, spec["preprocess"])


def build_classifier(cfg: TrainingConfig, num_classes: int) -> Model:
    """Backbone congelado + cabeza densa con regularización.

    La cabeza es deliberadamente la misma para cualquier backbone para
    que las comparaciones entre arquitecturas (B0 vs ResNet50V2 vs
    MobileNetV3) sean justas: la única variable que cambia es el
    extractor de características.
    """
    base_model = load_backbone(cfg, trainable=False)

    inputs = Input(shape=cfg.input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization()(x)

    for units, drop in zip(cfg.dense_units, cfg.dropout_rates):
        x = Dense(units, kernel_regularizer=l2(cfg.l2_reg))(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)

    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs, outputs, name=f"aquatic_classifier_{cfg.backbone}")
    model.base_model = base_model  # referencia directa para fine-tuning y Grad-CAM
    return model


def compile_model(model: Model, learning_rate: float, label_smoothing: float = 0.0) -> None:
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )


def unfreeze_for_finetuning(model: Model) -> None:
    """Descongela el backbone completo manteniendo BatchNorm en modo inferencia.

    Congelar las capas BatchNormalization durante el fine-tuning evita que
    sus estadísticas se desajusten con lotes pequeños, que es una causa
    común de que el fine-tuning "rompa" un modelo ya bueno.
    """
    model.base_model.trainable = True
    for layer in model.base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
