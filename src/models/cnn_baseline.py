"""Reusable CNN baseline for 0-255 RGB chest X-ray tensors."""

from __future__ import annotations

import tensorflow as tf


def build_cnn_baseline(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    """Build cnn_baseline_v1; input normalization from 0-255 is inside the model."""
    if len(input_shape) != 3 or input_shape[-1] != 3:
        raise ValueError("input_shape must be (height, width, 3)")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")

    inputs = tf.keras.Input(shape=input_shape, name="image_0_255")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)
    for index, filters in enumerate((32, 64, 128), start=1):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"block{index}_conv"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"block{index}_bn")(x)
        x = tf.keras.layers.ReLU(name=f"block{index}_relu")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"block{index}_pool")(x)

    x = tf.keras.layers.Conv2D(
        256, 3, padding="same", use_bias=False, name="block4_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block4_bn")(x)
    x = tf.keras.layers.ReLU(name="block4_relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)
    return tf.keras.Model(inputs, outputs, name="cnn_baseline_v1")
