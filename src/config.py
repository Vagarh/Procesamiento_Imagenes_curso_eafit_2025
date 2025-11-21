"""
Configuración centralizada para el experimento de detección de bounding boxes.

Este módulo contiene todos los hiperparámetros y rutas utilizadas
en el entrenamiento del modelo EfficientNetB3 para localización de objetos.
"""
from typing import Tuple

# ════════════════════════════════════════════════════════════════════════════
# Configuración del Modelo
# ════════════════════════════════════════════════════════════════════════════
IMG_SIZE: Tuple[int, int] = (300, 300)  # Tamaño de entrada para EfficientNetB3
BATCH_SIZE: int = 16                     # Tamaño del lote para entrenamiento

# ════════════════════════════════════════════════════════════════════════════
# Hiperparámetros de Entrenamiento
# ════════════════════════════════════════════════════════════════════════════
TOTAL_EPOCHS: int = 50   # Épocas totales para fine-tuning
LR_HEAD: float = 1e-3    # Learning rate para entrenamiento del head (fase 1)
LR_FINE: float = 1e-5    # Learning rate para fine-tuning (fase 2)
LAMBDA_IOU: float = 1.0  # Peso de la pérdida GIoU en la función de costo

# ════════════════════════════════════════════════════════════════════════════
# Rutas de Archivos
# ════════════════════════════════════════════════════════════════════════════
CSV_PATH: str = "data/Airplanes_clean.csv"   # Dataset de anotaciones
IMG_DIR: str = "data/airplanes"               # Directorio de imágenes
CHECKPOINT_DIR: str = "checkpoints"           # Directorio para guardar checkpoints
BEST_MODEL_PATH: str = "best_bbox_model.keras"  # Ruta del mejor modelo
