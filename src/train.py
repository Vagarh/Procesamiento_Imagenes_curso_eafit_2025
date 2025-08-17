import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras import callbacks

from src.data_loader import DataGenerator
from src.models import build_model
from src.losses import combined_loss
from src.visualization import plot_samples
from src.config import (
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

def main():
    """Main training script."""
    # 1) Carga y split
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 2) Generadores
    train_gen = DataGenerator(train_df, IMG_DIR, BATCH_SIZE, augment=True)
    val_gen   = DataGenerator(val_df,   IMG_DIR, BATCH_SIZE, augment=False)

    # 3) Construye modelo y captura backbone
    model, backbone = build_model()
    model.summary()

    # ---------- FASE 1 : solo head con LR alto ----------
    for layer in backbone.layers:
        layer.trainable = False

    decay_steps = len(train_gen) * 10  # 10 épocas
    lr_schedule = CosineDecay(LR_HEAD, decay_steps, alpha=1e-4)

    model.compile(optimizer=Adam(lr_schedule), loss=combined_loss(LAMBDA_IOU), metrics=['mae'])

    model.fit(
        train_gen, validation_data=val_gen,
        epochs=10,
        callbacks=[callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)]
    )
    plot_samples(model, val_df, IMG_DIR, n=5)

    # ---------- FASE 2 : fine-tune últimos bloques con LR bajo ----------
    for layer in backbone.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=Adam(LR_FINE), loss=combined_loss(LAMBDA_IOU), metrics=['mae'])

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=5),
        callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_loss'),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR,'epoch_{epoch:02d}.weights.h5'),
            save_weights_only=True, save_freq='epoch'
        )
    ]

    model.fit(
        train_gen, validation_data=val_gen,
        epochs=TOTAL_EPOCHS,
        callbacks=cb_list
    )

    # 5) Visualización final
    plot_samples(model, val_df, IMG_DIR, n=10)

if __name__ == '__main__':
    main()