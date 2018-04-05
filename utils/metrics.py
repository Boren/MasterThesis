#!/usr/bin/python

import tensorflow as tf
from keras import backend as keras_backend


def mean_iou(y_true, y_pred):
    """
    Uses the internal tensorflow calculations.
    This needs to be checked.

    """
    # TODO: Get number of classes dynamic
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, num_classes=10)
    keras_backend.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def dice_coefficient(y_true, y_pred):
    smooth = 1.

    y_true_f = keras_backend.flatten(y_true)
    y_pred_f = keras_backend.flatten(y_pred)
    intersection = keras_backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras_backend.sum(y_true_f) + keras_backend.sum(y_pred_f) + smooth)
