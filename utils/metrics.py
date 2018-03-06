#!/usr/bin/python

import tensorflow as tf
from keras import backend as K


def mean_iou(y_true, y_pred):
    """
    Uses the internal tensorflow calculations.
    This needs to be checked.

    """
    # TODO: Get number of classes dynamic
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, num_classes=10)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


