from keras import backend as keras_backend

from utils.metrics import dice_coefficient


def binary_crossentropy_with_logits(ground_truth, predictions):
    return keras_backend.mean(keras_backend.binary_crossentropy(ground_truth, predictions, from_logits=True), axis=-1)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)
