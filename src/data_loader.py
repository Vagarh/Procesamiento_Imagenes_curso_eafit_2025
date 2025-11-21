"""
Generador de datos para entrenamiento del modelo de detección.

Este módulo implementa un generador de datos compatible con Keras
que incluye aumentación de datos para mejorar la generalización.

Classes:
    DataGenerator: Generador de lotes con aumentación opcional
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tensorflow.keras.utils import Sequence

from .config import IMG_SIZE


class DataGenerator(Sequence):
    """
    Generador de datos compatible con Keras para detección de bounding boxes.

    Implementa la interfaz Sequence de Keras para generar lotes de datos
    de manera eficiente durante el entrenamiento. Soporta aumentación
    de datos mediante flip horizontal y variaciones de brillo.

    Attributes:
        df: DataFrame con anotaciones (Image, x_top, y_top, x_bottom, y_bottom)
        img_dir: Directorio con las imágenes
        batch_size: Tamaño del lote
        augment: Si aplicar aumentación de datos

    Example:
        >>> train_gen = DataGenerator(train_df, 'data/images', batch_size=16, augment=True)
        >>> val_gen = DataGenerator(val_df, 'data/images', batch_size=16, augment=False)
        >>> model.fit(train_gen, validation_data=val_gen)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        batch_size: int,
        augment: bool = False
    ) -> None:
        """
        Inicializa el generador de datos.

        Args:
            df: DataFrame con las anotaciones de bounding boxes
            img_dir: Ruta al directorio de imágenes
            batch_size: Número de muestras por lote
            augment: Si True, aplica aumentación de datos
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self) -> int:
        """Retorna el número de lotes por época."""
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera un lote de datos.

        Args:
            idx: Índice del lote

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - images: Array de forma (batch, H, W, 3) con valores en [0, 1]
                - boxes: Array de forma (batch, 4) con coordenadas normalizadas
        """
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.df))
        batch_df = self.df.iloc[start_idx:end_idx]

        images, boxes = [], []

        for _, row in batch_df.iterrows():
            # Cargar imagen
            img_path = os.path.join(self.img_dir, row.Image)
            img0 = Image.open(img_path).convert('RGB')
            w0, h0 = img0.size

            # Normalizar coordenadas de bounding box
            x1, x2 = sorted([row.x_top, row.x_bottom])
            y1, y2 = sorted([row.y_top, row.y_bottom])
            box = np.array([x1 / w0, y1 / h0, x2 / w0, y2 / h0], dtype=np.float32)

            # Redimensionar imagen
            img = img0.resize(IMG_SIZE)
            arr = np.array(img) / 255.0

            # Aumentación: flip horizontal
            if self.augment and np.random.rand() < 0.5:
                arr = np.fliplr(arr)
                box = np.array([1 - box[2], box[1], 1 - box[0], box[3]], dtype=np.float32)

            # Aumentación: variación de brillo
            if self.augment:
                brightness_factor = np.random.uniform(0.8, 1.2)
                img_pil = Image.fromarray((arr * 255).astype(np.uint8))
                img_bright = ImageEnhance.Brightness(img_pil).enhance(brightness_factor)
                arr = np.array(img_bright) / 255.0

            images.append(arr)
            boxes.append(box)

        return np.stack(images), np.stack(boxes)
