"""
Arquitectura del modelo para detección de bounding boxes.

Este módulo define la arquitectura del modelo basado en EfficientNetB3
con un head personalizado para regresión de bounding boxes.

Functions:
    build_model: Construye el modelo completo con backbone EfficientNetB3
"""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

from .losses import sort_corners
from .config import IMG_SIZE


def build_model() -> Tuple[models.Model, models.Model]:
    """
    Construye un modelo EfficientNetB3 para regresión de bounding boxes.

    Arquitectura:
        - Backbone: EfficientNetB3 pre-entrenado en ImageNet
        - Head: Dense(1024) -> Dropout(0.5) -> Dense(512) -> Dense(4, sigmoid)
        - Lambda layer para ordenar coordenadas de salida

    Returns:
        Tuple[Model, Model]: (modelo_completo, backbone)
            - modelo_completo: Modelo listo para entrenamiento
            - backbone: Backbone para control de capas congeladas

    Example:
        >>> model, backbone = build_model()
        >>> # Congelar backbone para primera fase
        >>> for layer in backbone.layers:
        ...     layer.trainable = False
        >>> model.compile(optimizer='adam', loss=combined_loss())
    """
    # Backbone pre-entrenado
    backbone = EfficientNetB3(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    # Head para regresión de bounding boxes
    x = layers.Dense(1024, activation='relu', name='head_dense1')(backbone.output)
    x = layers.Dropout(0.5, name='head_dropout')(x)
    x = layers.Dense(512, activation='relu', name='head_dense2')(x)
    raw = layers.Dense(4, activation='sigmoid', name='raw_bbox')(x)

    # Ordenar coordenadas para garantizar x1<=x2, y1<=y2
    ordered = layers.Lambda(sort_corners, name='ordered_bbox')(raw)

    model = models.Model(inputs=backbone.input, outputs=ordered, name='bbox_detector')
    return model, backbone
