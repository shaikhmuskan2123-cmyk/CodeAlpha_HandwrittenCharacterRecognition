"""Keras CNN architecture for the 26 EMNIST Letters classes."""

from __future__ import annotations


def create_model(input_shape: tuple[int, int, int] = (28, 28, 1)):
    """Build a compact CNN; TensorFlow is imported only when training is requested."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape, name="character_image")
    augmented = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.04, fill_mode="constant"),
            tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="constant"),
        ],
        name="training_augmentation",
    )(inputs)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(augmented)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(26, activation="softmax", name="class_probabilities")(x)

    model = tf.keras.Model(inputs, outputs, name="glyphsense_emnist_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model