"""Small custom CNN in TensorFlow/Keras (the baseline model)."""
import tensorflow as tf
from tensorflow.keras import layers

from .config import IMG_SIZE, TF_LAST_CONV_LAYER


def build_cnn() -> tf.keras.Model:
    """Three conv blocks, global average pooling, sigmoid output (probability of psoriasis).

    Rescaling and augmentation live inside the model, so serving passes raw 0-255 pixels.
    Augmentation layers are inactive at inference time.
    """
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomContrast(0.15)(x)

    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", name=TF_LAST_CONV_LAYER)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="psoriscan_cnn")
