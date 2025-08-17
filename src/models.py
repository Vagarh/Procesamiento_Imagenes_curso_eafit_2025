import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

from .losses import sort_corners # Import sort_corners from losses module
from .config import IMG_SIZE # Import IMG_SIZE from data_loader module

def build_model():
    backbone = EfficientNetB3(input_shape=IMG_SIZE+(3,), include_top=False,
                              weights='imagenet', pooling='avg')
    x = layers.Dense(1024, activation='relu')(backbone.output)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    raw = layers.Dense(4, activation='sigmoid')(x)
    ordered = layers.Lambda(sort_corners, name="ordered_bbox")(raw)
    return models.Model(backbone.input, ordered), backbone