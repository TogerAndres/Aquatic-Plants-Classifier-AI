"""Fija las semillas de aleatoriedad de todas las librerías relevantes.

Sin esto, dos corridas del mismo experimento (incluso con el mismo
config) pueden dar splits de K-Fold distintos y pesos iniciales
distintos, lo que hace imposible comparar resultados de forma justa.
"""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
