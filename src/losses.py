import tensorflow as tf
from tensorflow.keras.losses import Huber

huber = Huber()

def sort_corners(boxes):
    """Garantiza x1≤x2, y1≤y2 en un tensor [...,4]."""
    x1,y1,x2,y2 = tf.split(boxes,4,axis=-1)
    return tf.concat([tf.minimum(x1,x2), tf.minimum(y1,y2),
                      tf.maximum(x1,x2), tf.maximum(y1,y2)], axis=-1)

def giou_loss(y_true, y_pred):
    y_true, y_pred = sort_corners(y_true), sort_corners(y_pred)
    x1t,y1t,x2t,y2t = tf.split(y_true, 4, -1)
    x1p,y1p,x2p,y2p = tf.split(y_pred, 4, -1)

    xi1,yi1 = tf.maximum(x1t,x1p), tf.maximum(y1t,y1p)
    xi2,yi2 = tf.minimum(x2t,x2p), tf.minimum(y2t,y2p)
    inter   = tf.maximum(0., xi2-xi1) * tf.maximum(0., yi2-yi1)

    area_t = (x2t-x1t)*(y2t-y1t)
    area_p = (x2p-x1p)*(y2p-y1p)
    union  = area_t + area_p - inter + 1e-6
    iou    = inter / union

    cx1,cy1 = tf.minimum(x1t,x1p), tf.minimum(y1t,y1p)
    cx2,cy2 = tf.maximum(x2t,x2p), tf.maximum(y2t,y2p)
    area_c  = tf.maximum(0., cx2-cx1) * tf.maximum(0., cy2-cy1) + 1e-6

    giou = iou - (area_c - union) / area_c
    return tf.reduce_mean(1. - tf.clip_by_value(giou, -1., 1.))

def combined_loss(lambda_iou):
    def loss(y_true, y_pred):
        return huber(y_true, y_pred) + lambda_iou * giou_loss(y_true, y_pred)
    return loss