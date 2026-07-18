from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.models.transfer_models import (
    VGG16Preprocess,
    build_efficientnetb0_transfer,
    build_vgg16_transfer,
    freeze_backbone,
    get_backbone,
    unfreeze_efficientnet_last_non_bn,
    unfreeze_vgg16_block5,
)
from src.training.two_stage_training import compile_binary_model, resolve_stage_epochs
from src.train_transfer import create_transfer_run_directory


@pytest.fixture(scope="module")
def vgg():
    return build_vgg16_transfer((64, 64, 3), weights=None)


@pytest.fixture(scope="module")
def efficientnet():
    return build_efficientnetb0_transfer((64, 64, 3), weights=None)


def test_vgg_input_output_shape(vgg):
    assert vgg.input_shape == (None, 64, 64, 3) and vgg.output_shape == (None, 1)


def test_efficientnet_input_output_shape(efficientnet):
    assert efficientnet.input_shape == (None, 64, 64, 3) and efficientnet.output_shape == (None, 1)


@pytest.mark.parametrize("fixture_name", ["vgg", "efficientnet"])
def test_output_probability_range(request, fixture_name):
    model = request.getfixturevalue(fixture_name)
    values = model(tf.zeros((1, 64, 64, 3)), training=False).numpy()
    assert np.all((values >= 0) & (values <= 1))


@pytest.mark.parametrize("fixture_name", ["vgg", "efficientnet"])
def test_no_flatten(request, fixture_name):
    model = request.getfixturevalue(fixture_name)
    assert not any(isinstance(layer, tf.keras.layers.Flatten) for layer in model.layers)


def test_vgg_preprocess_is_canonical(vgg):
    layer = vgg.get_layer("vgg16_preprocess_input")
    assert isinstance(layer, VGG16Preprocess)
    sample = tf.constant([[[[1.0, 2.0, 3.0]]]])
    np.testing.assert_allclose(layer(sample), tf.keras.applications.vgg16.preprocess_input(sample))


def test_vgg_has_no_rescaling(vgg):
    assert not any(isinstance(layer, tf.keras.layers.Rescaling) for layer in vgg.layers)


def test_efficientnet_has_no_extra_top_level_rescaling(efficientnet):
    assert not any(isinstance(layer, tf.keras.layers.Rescaling) for layer in efficientnet.layers)


@pytest.mark.parametrize("fixture_name", ["vgg", "efficientnet"])
def test_phase1_backbone_frozen(request, fixture_name):
    model = request.getfixturevalue(fixture_name); freeze_backbone(model)
    assert not any(layer.trainable for layer in get_backbone(model).layers)


def test_vgg_phase2_only_block5(vgg):
    assert set(unfreeze_vgg16_block5(vgg)) == {"block5_conv1", "block5_conv2", "block5_conv3"}
    assert {layer.name for layer in get_backbone(vgg).layers if layer.trainable} == {
        "block5_conv1", "block5_conv2", "block5_conv3"
    }


def test_efficientnet_phase2_keeps_batch_norm_frozen(efficientnet):
    unfrozen = unfreeze_efficientnet_last_non_bn(efficientnet, 20)
    assert len(unfrozen) <= 20
    assert all(not layer.trainable for layer in get_backbone(efficientnet).layers
               if isinstance(layer, tf.keras.layers.BatchNormalization))


def test_efficientnet_backbone_called_in_inference_mode():
    source = (Path(__file__).resolve().parents[1] / "src/models/transfer_models.py").read_text()
    assert "backbone(inputs, training=False)" in source


def test_recompile_creates_new_optimizer(vgg):
    compile_binary_model(vgg, 1e-3); first = vgg.optimizer
    compile_binary_model(vgg, 1e-5)
    assert vgg.optimizer is not first


def test_pilot_epoch_override():
    assert resolve_stage_epochs({}, "pilot") == (1, 1)


def test_output_directory_no_overwrite(tmp_path):
    config = {"experiment_name": "vgg16_transfer_v1"}
    create_transfer_run_directory(config, "pilot", tmp_path, "same")
    with pytest.raises(FileExistsError):
        create_transfer_run_directory(config, "pilot", tmp_path, "same")


def test_training_entry_does_not_access_test_manifest():
    source = (Path(__file__).resolve().parents[1] / "src/train_transfer.py").read_text()
    assert '["test_manifest"]' not in source and "['test_manifest']" not in source
