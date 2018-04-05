from keras import backend as keras_backend


def binary_crossentropy_with_logits(ground_truth, predictions):
    return keras_backend.mean(keras_backend.binary_crossentropy(ground_truth, predictions, from_logits=True), axis=-1)
