from typing import Tuple

import keras.models as models
from keras import Model
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2

from utils import metrics


def tiramisu(input_size: int, num_classes: int, channels: int = 3) -> Tuple[Model, str]:
    """
    The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
    https://arxiv.org/abs/1611.09326
    https://github.com/0bserver07/One-Hundred-Layers-Tiramisu

    103 layers
    """
    if input_size <= 8:
        raise Exception("Input size too small")

    model = models.Sequential()

    model.add(Conv2D(48, kernel_size=(3, 3), padding='same',
                     input_shape=(input_size, input_size, channels),
                     kernel_initializer="he_uniform",
                     kernel_regularizer=l2(0.0001),
                     data_format='channels_last'))

    model = dense_block(model, 4, 112)
    model = transition_down(model, 112)
    model = dense_block(model, 5, 192)
    model = transition_down(model, 192)
    model = dense_block(model, 7, 304)
    model = transition_down(model, 304)
    model = dense_block(model, 10, 464)
    model = transition_down(model, 464)
    model = dense_block(model, 12, 656)
    model = transition_down(model, 656)

    model = dense_block(model, 15, 896)

    model = transition_up(model, 1088, (1088, 7, 7))
    model = dense_block(model, 12, 1088)

    model = transition_up(model, 816, (816, 14, 14))
    model = dense_block(model, 10, 816)

    model = transition_up(model, 576, (576, 28, 28))
    model = dense_block(model, 7, 576)

    model = transition_up(model, 384, (384, 56, 56))
    model = dense_block(model, 5, 384)

    model = transition_up(model, 256, (256, 112, 112))
    model = dense_block(model, 4, 256)

    model.add(Conv2D(num_classes, kernel_size=(1, 1),
                     padding='same',
                     kernel_initializer="he_uniform",
                     kernel_regularizer=l2(0.0001),
                     data_format='channels_last'))

    model.add(Conv2D(num_classes, (1, 1), activation='sigmoid'))

    model.compile(optimizer=Adam(lr=1e-3, decay=0.995),
                  loss='binary_crossentropy',
                  metrics=['accuracy', metrics.mean_iou])

    return model, "tiramisu"


def dense_block(model, layers, filters):
    for i in range(layers):
        model.add(BatchNormalization(axis=1,
                                     gamma_regularizer=l2(0.0001),
                                     beta_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(Conv2D(filters, kernel_size=(3, 3), padding='same',
                         kernel_initializer="he_uniform",
                         data_format='channels_last'))
        model.add(Dropout(0.2))

    return model


def transition_down(model, filters):
    model.add(BatchNormalization(axis=1,
                                 gamma_regularizer=l2(0.0001),
                                 beta_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters, kernel_size=(1, 1), padding='same',
                     kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           data_format='channels_last'))

    return model


def transition_up(model, filters, input_shape):
    model.add(Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2),
                              padding='same',
                              input_shape=input_shape,
                              kernel_initializer="he_uniform",
                              data_format='channels_last'))

    return model
