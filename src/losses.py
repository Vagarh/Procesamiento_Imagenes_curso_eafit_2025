"""
Funciones de pérdida para detección de bounding boxes.

Este módulo implementa funciones de pérdida especializadas para tareas
de localización de objetos, incluyendo GIoU (Generalized IoU) loss.

Functions:
    sort_corners: Ordena las coordenadas para garantizar x1<=x2, y1<=y2
    giou_loss: Calcula la pérdida Generalized IoU
    combined_loss: Combina pérdida Huber con GIoU loss
"""
from typing import Callable

import tensorflow as tf
from tensorflow.keras.losses import Huber

# Instancia global de pérdida Huber
huber = Huber()


def sort_corners(boxes: tf.Tensor) -> tf.Tensor:
    """
    Garantiza que las coordenadas estén ordenadas: x1<=x2, y1<=y2.

    Args:
        boxes: Tensor de forma [..., 4] con coordenadas (x1, y1, x2, y2)

    Returns:
        Tensor con coordenadas ordenadas de la misma forma que la entrada
    """
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    return tf.concat([
        tf.minimum(x1, x2),
        tf.minimum(y1, y2),
        tf.maximum(x1, x2),
        tf.maximum(y1, y2)
    ], axis=-1)


def giou_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calcula la pérdida Generalized IoU entre bounding boxes.

    GIoU es una métrica mejorada sobre IoU que considera también el área
    de la caja envolvente, permitiendo gradientes incluso cuando las cajas
    no se superponen.

    Args:
        y_true: Tensor de ground truth boxes, forma (batch, 4)
        y_pred: Tensor de predicciones, forma (batch, 4)

    Returns:
        Escalar con el valor medio de la pérdida GIoU (1 - GIoU)

    Reference:
        Rezatofighi et al. "Generalized Intersection over Union" (CVPR 2019)
    """
    y_true = sort_corners(y_true)
    y_pred = sort_corners(y_pred)

    # Extraer coordenadas
    x1t, y1t, x2t, y2t = tf.split(y_true, 4, axis=-1)
    x1p, y1p, x2p, y2p = tf.split(y_pred, 4, axis=-1)

    # Calcular intersección
    xi1 = tf.maximum(x1t, x1p)
    yi1 = tf.maximum(y1t, y1p)
    xi2 = tf.minimum(x2t, x2p)
    yi2 = tf.minimum(y2t, y2p)
    inter = tf.maximum(0.0, xi2 - xi1) * tf.maximum(0.0, yi2 - yi1)

    # Calcular áreas y unión
    area_t = (x2t - x1t) * (y2t - y1t)
    area_p = (x2p - x1p) * (y2p - y1p)
    union = area_t + area_p - inter + 1e-6

    # IoU
    iou = inter / union

    # Caja envolvente C
    cx1 = tf.minimum(x1t, x1p)
    cy1 = tf.minimum(y1t, y1p)
    cx2 = tf.maximum(x2t, x2p)
    cy2 = tf.maximum(y2t, y2p)
    area_c = tf.maximum(0.0, cx2 - cx1) * tf.maximum(0.0, cy2 - cy1) + 1e-6

    # GIoU y pérdida
    giou = iou - (area_c - union) / area_c
    return tf.reduce_mean(1.0 - tf.clip_by_value(giou, -1.0, 1.0))


def combined_loss(lambda_iou: float = 1.0) -> Callable:
    """
    Crea una función de pérdida combinada: Huber + λ * GIoU.

    La pérdida Huber proporciona estabilidad para regresión, mientras que
    GIoU asegura que las cajas predichas se alineen con el ground truth.

    Args:
        lambda_iou: Peso de la pérdida GIoU (default: 1.0)

    Returns:
        Función de pérdida que acepta (y_true, y_pred) -> loss

    Example:
        >>> loss_fn = combined_loss(lambda_iou=1.0)
        >>> model.compile(loss=loss_fn)
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return huber(y_true, y_pred) + lambda_iou * giou_loss(y_true, y_pred)
    return loss
