"""Gráficas reutilizables para reportes de entrenamiento y evaluación."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # backend sin display, necesario para correr en servidores/CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_curves(history_head, history_finetune, save_path: str) -> None:
    acc = history_head.history["accuracy"] + history_finetune.history["accuracy"]
    val_acc = history_head.history["val_accuracy"] + history_finetune.history["val_accuracy"]
    loss = history_head.history["loss"] + history_finetune.history["loss"]
    val_loss = history_head.history["val_loss"] + history_finetune.history["val_loss"]
    split_epoch = len(history_head.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(acc, label="Train")
    axes[0].plot(val_acc, label="Val")
    axes[0].axvline(split_epoch, color="green", linestyle="--", alpha=0.7, label="Inicio fine-tuning")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Época")
    axes[0].legend()

    axes[1].plot(loss, label="Train")
    axes[1].plot(val_loss, label="Val")
    axes[1].axvline(split_epoch, color="green", linestyle="--", alpha=0.7, label="Inicio fine-tuning")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Época")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="mako",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión normalizada")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
