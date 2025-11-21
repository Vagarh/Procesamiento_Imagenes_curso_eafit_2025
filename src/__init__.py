"""
Paquete src - Módulos para detección de bounding boxes en imágenes.

Curso: Procesamiento de Imágenes - EAFIT 2025
Descripción: Sistema de detección de objetos usando EfficientNetB3 con
             pérdida combinada Huber + GIoU para localización de bounding boxes.

Módulos:
    - config: Configuración de hiperparámetros y rutas
    - data_loader: Generador de datos con aumentación
    - models: Arquitectura del modelo EfficientNetB3
    - losses: Funciones de pérdida (GIoU, Huber combinado)
    - visualization: Utilidades de visualización
    - train: Script principal de entrenamiento
"""

from .config import (
    IMG_SIZE,
    BATCH_SIZE,
    TOTAL_EPOCHS,
    LR_HEAD,
    LR_FINE,
    LAMBDA_IOU,
    CSV_PATH,
    IMG_DIR,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
)
from .data_loader import DataGenerator
from .models import build_model
from .losses import sort_corners, giou_loss, combined_loss
from .visualization import plot_samples

__version__ = "1.0.0"
__author__ = "EAFIT - Maestría en Ciencia de Datos"

__all__ = [
    # Config
    "IMG_SIZE",
    "BATCH_SIZE",
    "TOTAL_EPOCHS",
    "LR_HEAD",
    "LR_FINE",
    "LAMBDA_IOU",
    "CSV_PATH",
    "IMG_DIR",
    "CHECKPOINT_DIR",
    "BEST_MODEL_PATH",
    # Classes
    "DataGenerator",
    # Functions
    "build_model",
    "sort_corners",
    "giou_loss",
    "combined_loss",
    "plot_samples",
]
