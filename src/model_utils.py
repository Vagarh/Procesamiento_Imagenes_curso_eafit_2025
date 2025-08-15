import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import Huber

# Constants (can be moved to a config file later if needed)
IMG_SIZE = (300, 300)
LAMBDA_IOU = 1.0

def sort_corners(boxes):
    """
    Asegura x1<=x2, y1<=y2 en un tensor [...,4]
    boxes = [..., (x1,y1,x2,y2)]
    """
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    new_x1 = tf.minimum(x1, x2)
    new_x2 = tf.maximum(x1, x2)
    new_y1 = tf.minimum(y1, y2)
    new_y2 = tf.maximum(y1, y2)
    return tf.concat([new_x1, new_y1, new_x2, new_y2], axis=-1)

huber = Huber()

def giou_loss(y_true, y_pred):
    y_true = sort_corners(y_true)
    y_pred = sort_corners(y_pred)

    # 1) Divide en coordenadas
    x1_t,y1_t,x2_t,y2_t = tf.split(y_true, 4, axis=-1)
    x1_p,y1_p,x2_p,y2_p = tf.split(y_pred, 4, axis=-1)

    # 2) Intersección
    xi1 = tf.maximum(x1_t, x1_p)
    yi1 = tf.maximum(y1_t, y1_p)
    xi2 = tf.minimum(x2_t, x2_p)
    yi2 = tf.minimum(y2_t, y2_p)
    i_w  = tf.maximum(0.0, xi2 - xi1)
    i_h  = tf.maximum(0.0, yi2 - yi1)
    inter = i_w * i_h

    # 3) Áreas
    area_t = (x2_t-x1_t)*(y2_t-y1_t)
    area_p = (x2_p-x1_p)*(y2_p-y1_p)
    union  = area_t + area_p - inter + 1e-6

    # 4) IoU
    iou = inter / union

    # 5) Caja envolvente C
    c_x1 = tf.minimum(x1_t, x1_p)
    c_y1 = tf.minimum(y1_t, y1_p)
    c_x2 = tf.maximum(x2_t, x2_p)
    c_y2 = tf.maximum(y2_t, y2_p)
    c_w  = tf.maximum(0.0, c_x2 - c_x1)
    c_h  = tf.maximum(0.0, c_y2 - c_y1)
    area_c = c_w * c_h + 1e-6

    # 6) GIoU y loss
    giou = iou - (area_c - union) / area_c
    giou = tf.clip_by_value(giou, -1.0, 1.0)
    loss = 1.0 - giou
    loss = tf.maximum(loss, 0.0)
    return tf.reduce_mean(loss)

def combined_loss(y_true, y_pred):
    return huber(y_true, y_pred) + LAMBDA_IOU * giou_loss(y_true, y_pred)

def build_model():
    backbone = models.EfficientNetB3(
        input_shape=IMG_SIZE+(3,),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    x = layers.Dense(1024, activation='relu')(backbone.output)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    out = layers.Dense(4, activation='sigmoid')(x)
    model = models.Model(inputs=backbone.input, outputs=out)
    return model, backbone