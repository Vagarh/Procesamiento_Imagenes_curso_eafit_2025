import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras import callbacks

# Import modules
from data_loader import DataGenerator
from model_utils import build_model, combined_loss, IMG_SIZE, LAMBDA_IOU
from visualization import plot_samples

# Configuration (can be moved to a separate config file later)
BATCH_SIZE = 16
TOTAL_EPOCHS = 50
LR_HEAD = 1e-3
LR_FINE = 1e-5

# Update paths to be relative to the project root
PROJECT_ROOT = "d:\\Users\\jcardonr\\Documents\\Procesamiento_Imagenes_curso_eafit_2025"
CSV_PATH = os.path.join(PROJECT_ROOT, "legacy", "Taller_05_Deteccion_Realizar un modelo de deteccion de boundboxes_ok", "Airplanes_clean.csv")
IMG_DIR = os.path.join(PROJECT_ROOT, "legacy", "Taller_05_Deteccion_Realizar un modelo de deteccion de boundboxes_ok", "airplanes")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_bbox_model.keras")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if __name__ == '__main__':
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

    model.compile(optimizer=Adam(lr_schedule), loss=combined_loss, metrics=['mae'])

    history_head = model.fit(
        train_gen, validation_data=val_gen,
        epochs=10,
        callbacks=[callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)]
    )
    plot_samples(model, val_df, IMG_DIR, n=5)

    # ---------- FASE 2 : fine-tune últimos bloques con LR bajo ----------
    for layer in backbone.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=Adam(LR_FINE), loss=combined_loss, metrics=['mae'])

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
