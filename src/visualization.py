"""
Utilidades de visualización para el modelo de detección.

Este módulo proporciona funciones para visualizar las predicciones
del modelo comparadas con el ground truth.

Functions:
    plot_samples: Visualiza predicciones vs ground truth con IoU
"""
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tensorflow.keras import Model

from .losses import sort_corners
from .config import IMG_SIZE  # Corregido: importar desde config, no data_loader


def plot_samples(
    model: Model,
    df: pd.DataFrame,
    img_dir: str,
    n: int = 5
) -> None:
    """
    Visualiza n muestras aleatorias con predicciones vs ground truth.

    Para cada imagen seleccionada, muestra:
    - Bounding box ground truth (azul)
    - Bounding box predicho (rojo)
    - Valor de IoU en el título

    Args:
        model: Modelo Keras entrenado para predicción de bounding boxes
        df: DataFrame con columnas 'Image', 'x_top', 'y_top', 'x_bottom', 'y_bottom'
        img_dir: Directorio que contiene las imágenes
        n: Número de muestras a visualizar (default: 5)

    Example:
        >>> from src.visualization import plot_samples
        >>> plot_samples(model, val_df, 'data/airplanes', n=5)
    """
    sample = df.sample(n).reset_index(drop=True)

    for _, row in sample.iterrows():
        # Cargar y preprocesar imagen
        img_path = os.path.join(img_dir, row.Image)
        img0 = Image.open(img_path).convert('RGB')
        w0, h0 = img0.size
        img = img0.resize(IMG_SIZE)
        arr = np.array(img) / 255.0

        # Obtener predicción
        pred = model.predict(arr[np.newaxis], verbose=0)[0]
        pred = sort_corners(pred).numpy()

        # Calcular ground truth normalizado
        x1, x2 = sorted([row.x_top, row.x_bottom])
        y1, y2 = sorted([row.y_top, row.y_bottom])
        gt = np.array([x1 / w0, y1 / h0, x2 / w0, y2 / h0])

        # Escalar a coordenadas de imagen redimensionada
        scale = np.array([IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1]])
        x1g, y1g, x2g, y2g = gt * scale
        x1p, y1p, x2p, y2p = pred * scale

        # Calcular IoU
        inter = (
            max(0, min(x2g, x2p) - max(x1g, x1p)) *
            max(0, min(y2g, y2p) - max(y1g, y1p))
        )
        union = (x2g - x1g) * (y2g - y1g) + (x2p - x1p) * (y2p - y1p) - inter + 1e-6
        iou = inter / union

        # Visualizar
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        ax = plt.gca()

        # Ground truth (azul)
        ax.add_patch(plt.Rectangle(
            (x1g, y1g), x2g - x1g, y2g - y1g,
            edgecolor='blue', fill=False, linewidth=2, label='Ground Truth'
        ))

        # Predicción (rojo)
        ax.add_patch(plt.Rectangle(
            (x1p, y1p), x2p - x1p, y2p - y1p,
            edgecolor='red', fill=False, linewidth=2, label='Predicción'
        ))

        plt.title(f"IoU: {iou:.3f}")
        plt.axis('off')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
