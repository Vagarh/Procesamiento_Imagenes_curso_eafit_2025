"""
Configuration for the experiment.
"""

# ╔═══════════ Configuración ═══════════╗
IMG_SIZE      = (300, 300)
BATCH_SIZE    = 16
TOTAL_EPOCHS  = 50
LR_HEAD       = 1e-3
LR_FINE       = 1e-5
LAMBDA_IOU    = 1.0

CSV_PATH = "data/Airplanes_clean.csv"
IMG_DIR  = "data/airplanes"
CHECKPOINT_DIR  = 'checkpoints'
BEST_MODEL_PATH = 'best_bbox_model.keras'
