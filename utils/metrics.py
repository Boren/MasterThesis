#!/usr/bin/python

from keras import backend as keras_backend
import tensorflow as tf


def dice_coefficient(y_true, y_pred):
    smooth = 1.

    y_true_f = keras_backend.flatten(y_true)
    y_pred_f = keras_backend.flatten(y_pred)
    intersection = keras_backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
            keras_backend.sum(y_true_f) + keras_backend.sum(
        y_pred_f) + smooth)


def iou(y_true, y_pred):
    s = keras_backend.shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_true_reshaped = keras_backend.reshape(y_true,
                                            tf.stack([-1, s[1] * s[2], s[-1]]))
    y_pred_reshaped = keras_backend.reshape(y_pred,
                                            tf.stack([-1, s[1] * s[2], s[-1]]))

    # correctly classified
    clf_pred = keras_backend.one_hot(keras_backend.argmax(y_pred_reshaped),
                                     nb_classes=s[-1])
    equal_entries = keras_backend.cast(
        keras_backend.equal(clf_pred, y_true_reshaped),
        dtype='float32') * y_true_reshaped

    intersection = keras_backend.sum(equal_entries, axis=1)
    union_per_class = keras_backend.sum(y_true_reshaped,
                                        axis=1) + keras_backend.sum(
        y_pred_reshaped,
        axis=1)

    iou = intersection / (union_per_class - intersection)
    iou_mask = tf.is_finite(iou)
    iou_masked = tf.boolean_mask(iou, iou_mask)

    print(iou_masked)

    return keras_backend.mean(iou_masked)
