"""
Script principal de entrenamiento para el modelo de detección de bounding boxes.

Este script implementa un entrenamiento en dos fases:
    1. Entrenamiento del head con backbone congelado (LR alto)
    2. Fine-tuning de las últimas capas del backbone (LR bajo)

Usage:
    python -m src.train

    O desde el directorio raíz:
    python src/train.py
"""
import os
import logging
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras import callbacks

from .data_loader import DataGenerator
from .models import build_model
from .losses import combined_loss
from .visualization import plot_samples
from .config import (
    CSV_PATH,
    IMG_DIR,
    BATCH_SIZE,
    TOTAL_EPOCHS,
    LR_HEAD,
    LR_FINE,
    LAMBDA_IOU,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(csv_path: Optional[str] = None, img_dir: Optional[str] = None) -> None:
    """
    Ejecuta el pipeline completo de entrenamiento.

    El entrenamiento se realiza en dos fases:
        1. Fase Head: 10 épocas con backbone congelado y LR=1e-3
        2. Fase Fine-tune: TOTAL_EPOCHS épocas con últimas 20 capas descongeladas

    Args:
        csv_path: Ruta al archivo CSV con anotaciones (usa config si None)
        img_dir: Directorio de imágenes (usa config si None)

    Raises:
        FileNotFoundError: Si el CSV o directorio de imágenes no existe
    """
    # Usar valores de config si no se especifican
    csv_path = csv_path or CSV_PATH
    img_dir = img_dir or IMG_DIR

    # Validar que existen los archivos
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Directorio de imágenes no encontrado: {img_dir}")

    logger.info("Iniciando entrenamiento...")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Imágenes: {img_dir}")

    # 1) Cargar datos y dividir
    logger.info("Cargando dataset...")
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Muestras - Train: {len(train_df)}, Val: {len(val_df)}")

    # 2) Crear generadores
    train_gen = DataGenerator(train_df, img_dir, BATCH_SIZE, augment=True)
    val_gen = DataGenerator(val_df, img_dir, BATCH_SIZE, augment=False)

    # 3) Construir modelo
    logger.info("Construyendo modelo...")
    model, backbone = build_model()
    model.summary(print_fn=logger.info)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1: Entrenamiento del Head (backbone congelado)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("FASE 1: Entrenando head con backbone congelado...")
    for layer in backbone.layers:
        layer.trainable = False

    decay_steps = len(train_gen) * 10  # 10 épocas
    lr_schedule = CosineDecay(LR_HEAD, decay_steps, alpha=1e-4)

    model.compile(
        optimizer=Adam(lr_schedule),
        loss=combined_loss(LAMBDA_IOU),
        metrics=['mae']
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[
            callbacks.EarlyStopping(
                'val_loss', patience=3, restore_best_weights=True
            )
        ]
    )

    logger.info("Visualizando resultados de Fase 1...")
    plot_samples(model, val_df, img_dir, n=5)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2: Fine-tuning (últimas capas descongeladas)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("FASE 2: Fine-tuning de últimas 20 capas del backbone...")
    for layer in backbone.layers[-20:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(LR_FINE),
        loss=combined_loss(LAMBDA_IOU),
        metrics=['mae']
    )

    # Crear directorio de checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            'val_loss', patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=5),
        callbacks.ModelCheckpoint(
            BEST_MODEL_PATH, save_best_only=True, monitor='val_loss'
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, 'epoch_{epoch:02d}.weights.h5'),
            save_weights_only=True,
            save_freq='epoch'
        )
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=TOTAL_EPOCHS,
        callbacks=cb_list
    )

    # 4) Visualización final
    logger.info("Entrenamiento completado. Visualizando resultados finales...")
    plot_samples(model, val_df, img_dir, n=10)


if __name__ == '__main__':
    main()
