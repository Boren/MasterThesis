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
