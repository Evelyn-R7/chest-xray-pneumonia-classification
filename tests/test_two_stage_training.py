import pandas as pd
import tensorflow as tf

from src.training.two_stage_training import (
    build_phase_callbacks,
    run_two_stage_training,
)


def tiny_model():
    inputs = tf.keras.Input((4,))
    backbone_inputs = tf.keras.Input((4,))
    x = tf.keras.layers.Dense(3, activation="relu", name="feature_dense")(backbone_inputs)
    backbone = tf.keras.Model(backbone_inputs, x, name="vgg16_backbone")
    backbone.trainable = False
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(backbone(inputs, training=False))
    return tf.keras.Model(inputs, outputs)


def config():
    return {
        "head_learning_rate": 0.001, "fine_tune_learning_rate": 0.00001,
        "head_max_epochs": 1, "fine_tune_max_epochs": 1,
        "head_early_stopping_patience": 3, "fine_tune_early_stopping_patience": 5,
        "reduce_lr_patience": 2, "reduce_lr_factor": 0.5, "minimum_learning_rate": 1e-6,
    }


def freeze(model):
    backbone = model.get_layer("vgg16_backbone"); backbone.trainable = False
    for layer in backbone.layers: layer.trainable = False
    return []


def unfreeze(model):
    backbone = model.get_layer("vgg16_backbone"); backbone.trainable = True
    layer = backbone.get_layer("feature_dense"); layer.trainable = True
    return [layer.name]


def test_phase_callbacks_are_fresh_instances(tmp_path):
    first = build_phase_callbacks(tmp_path, "phase1", 3, 2, 0.5, 1e-6)
    second = build_phase_callbacks(tmp_path, "phase2", 5, 2, 0.5, 1e-6)
    assert all(a is not b for a, b in zip(first, second))


def test_two_stage_synthetic_training(tmp_path):
    x = tf.random.uniform((16, 4), seed=1); y = tf.cast(tf.reduce_sum(x, axis=1) > 2, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
    result = run_two_stage_training(tiny_model(), dataset, dataset, tmp_path, config(), "pilot", freeze, unfreeze)
    assert result["best_phase"] in {"phase1", "phase2"}
    assert result["unfrozen_layers"] == ["feature_dense"]
    assert (tmp_path / "best_model.keras").is_file()
    phase1 = pd.read_csv(tmp_path / "phase1_history.csv")
    phase2 = pd.read_csv(tmp_path / "phase2_history.csv")
    assert phase1.loc[0, "epoch"] == 0 and phase2.loc[0, "epoch"] == 1
    assert len(pd.read_csv(tmp_path / "combined_history.csv")) == 2
