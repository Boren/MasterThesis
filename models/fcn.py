from typing import Tuple

import tensorflow as tf

from keras import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, LeakyReLU
from keras.optimizers import Adam

from .utils import BilinearUpSampling2D


def fcn32(input_size: int, num_classes: int, channels: int = 3) -> Tuple[Model, str]:
    """
    Fully Convolutional Networks for Semantic Segmentation

    https://github.com/divamgupta/image-segmentation-keras
    """
    img_input = Input((input_size, input_size, channels))

    # BLOCK 1
    with tf.name_scope('Block_1'):
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
        x = LeakyReLU()(x)
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # BLOCK 2
    with tf.name_scope('Block_2'):
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
        x = LeakyReLU()(x)
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # BLOCK 3
    with tf.name_scope('Block_3'):
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # BLOCK 4
    with tf.name_scope('Block_4'):
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # BLOCK 5
    with tf.name_scope('Block_5'):
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Conv Layers -> Fully-Connected Layers
    with tf.name_scope('Fully_Connected'):
        x = Conv2D(4096, (7, 7), padding='same', name='fc1')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(4096, (1, 1), padding='same', name='fc2')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.5)(x)

    # CLASSIFIER
    with tf.name_scope('Classification'):
        x = Conv2D(num_classes, (1, 1), padding='valid', strides=(1, 1),
                   name="Softmax_Activation", activation='softmax')(x)

    x = BilinearUpSampling2D(size=(32, 32), name="Upsampling")(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model, "fcn"
