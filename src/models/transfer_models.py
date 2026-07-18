"""VGG16 and EfficientNetB0 transfer-learning model builders."""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="chest_xray")
class VGG16Preprocess(tf.keras.layers.Layer):
    """Apply the canonical VGG16 preprocessing to 0-255 RGB input."""

    def call(self, inputs):
        return tf.keras.applications.vgg16.preprocess_input(inputs)


def _classification_head(features, dropout_rate: float):
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    return tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)


def build_vgg16_transfer(
    input_shape=(224, 224, 3), dropout_rate: float = 0.4, weights: str | None = "imagenet",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="image_0_255")
    preprocessed = VGG16Preprocess(name="vgg16_preprocess_input")(inputs)
    backbone = tf.keras.applications.VGG16(
        include_top=False, weights=weights, input_shape=input_shape, name="vgg16_backbone"
    )
    backbone.trainable = False
    outputs = _classification_head(backbone(preprocessed, training=False), dropout_rate)
    return tf.keras.Model(inputs, outputs, name="vgg16_transfer_v1")


def build_efficientnetb0_transfer(
    input_shape=(224, 224, 3), dropout_rate: float = 0.4, weights: str | None = "imagenet",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="image_0_255")
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=input_shape, name="efficientnetb0"
    )
    backbone.trainable = False
    # Inference mode is intentional so BatchNormalization moving statistics never update.
    outputs = _classification_head(backbone(inputs, training=False), dropout_rate)
    return tf.keras.Model(inputs, outputs, name="efficientnetb0_transfer_v1")


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for name in ("vgg16_backbone", "efficientnetb0"):
        try:
            return model.get_layer(name)
        except ValueError:
            pass
    raise ValueError("Transfer model backbone not found")


def freeze_backbone(model: tf.keras.Model) -> list[str]:
    backbone = get_backbone(model)
    backbone.trainable = False
    for layer in backbone.layers:
        layer.trainable = False
    return []


def unfreeze_vgg16_block5(model: tf.keras.Model) -> list[str]:
    backbone = get_backbone(model)
    allowed = {"block5_conv1", "block5_conv2", "block5_conv3"}
    backbone.trainable = True
    unfrozen = []
    for layer in backbone.layers:
        layer.trainable = layer.name in allowed
        if layer.trainable:
            unfrozen.append(layer.name)
    if set(unfrozen) != allowed:
        raise RuntimeError(f"Unexpected VGG16 block5 layers: {unfrozen}")
    return unfrozen


def unfreeze_efficientnet_last_non_bn(
    model: tf.keras.Model, count: int = 20
) -> list[str]:
    backbone = get_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    candidates = [
        layer for layer in reversed(backbone.layers)
        if not isinstance(layer, tf.keras.layers.BatchNormalization)
        and layer.weights
    ][:count]
    selected = set(candidates)
    unfrozen = []
    for layer in backbone.layers:
        layer.trainable = layer in selected
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        elif layer.trainable:
            unfrozen.append(layer.name)
    return unfrozen


def trainable_parameter_count(model: tf.keras.Model) -> int:
    return int(sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights))
