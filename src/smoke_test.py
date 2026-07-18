from pathlib import Path

import tensorflow as tf

from data_pipeline import build_all_datasets


def main():
    root = Path(__file__).resolve().parents[1]
    train, val, _, _ = build_all_datasets(root / "configs" / "data_v3_clean.yaml")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(224, 224, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(8, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train, validation_data=val, epochs=1, steps_per_epoch=2,
                        validation_steps=1, verbose=2)
    values = {key: float(value[-1]) for key, value in history.history.items()}
    assert all(tf.math.is_finite(value) for value in values.values())
    print({"status": "success", "formal_metric": False, "steps_per_epoch": 2,
           "validation_steps": 1, "history": values})
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
