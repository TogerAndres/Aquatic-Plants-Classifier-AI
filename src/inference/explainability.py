from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


class GradCAM:

    def __init__(self, model: tf.keras.Model, cfg=None):
        self.model = model
        self.cfg = cfg

        self.last_conv_layer = self._find_last_conv_layer()

        self.grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.last_conv_layer.output,
                self.model.output,
            ],
        )

    def _find_last_conv_layer(self):

        for layer in reversed(self.model.layers):

            try:

                shape = layer.output.shape

                if len(shape) == 4:
                    return layer

            except Exception:
                pass

        raise RuntimeError("No se encontró una capa convolucional.")

    def compute_heatmap(
        self,
        image,
        class_index=None,
    ):

        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        image = tf.cast(image, tf.float32)

        with tf.GradientTape() as tape:

            conv_outputs, predictions = self.grad_model(image)

            if class_index is None:
                class_index = tf.argmax(predictions[0])

            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2),
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1,
        )

        heatmap = tf.maximum(heatmap, 0)

        max_heat = tf.reduce_max(heatmap)

        if max_heat > 0:
            heatmap /= max_heat

        return heatmap.numpy(), int(class_index)

    def overlay_heatmap(
        self,
        image,
        heatmap,
        alpha=0.45,
    ):

        h, w = image.shape[:2]

        heatmap = cv2.resize(
            heatmap,
            (w, h),
        )

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET,
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB,
        )

        overlay = cv2.addWeighted(
            image,
            1 - alpha,
            heatmap,
            alpha,
            0,
        )

        return overlay

    def region_focus_summary(self, heatmap):

        h, w = heatmap.shape

        regions = {
            "centro": heatmap[h//3:2*h//3, w//3:2*w//3],
            "borde_superior": heatmap[:h//3, :],
            "borde_inferior": heatmap[2*h//3:, :],
            "borde_izquierdo": heatmap[:, :w//3],
            "borde_derecho": heatmap[:, 2*w//3:],
        }

        total = float(np.sum(heatmap))

        if total == 0:
            total = 1

        scores = {
            k: float(np.sum(v))/total
            for k, v in regions.items()
        }

        dominant = max(scores, key=scores.get)

        return {
            "region_scores": scores,
            "dominant_region": dominant,
        }