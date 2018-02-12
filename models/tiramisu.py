from __future__ import absolute_import
from __future__ import print_function
import os

import keras.models as models

from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose
from keras.regularizers import l2
from keras import backend as K

K.set_image_dim_ordering('tf')

# weight_decay = 0.0001


class Tiramisu:
    """
    The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
    https://arxiv.org/abs/1611.09326
    https://github.com/0bserver07/One-Hundred-Layers-Tiramisu

    103 layers
    """
    def __init__(self):
        self.create()

    def dense_block(self, layers, filters):

        model = self.model
        for i in range(layers):
            model.add(BatchNormalization(mode=0, axis=1,
                                         gamma_regularizer=l2(0.0001),
                                         beta_regularizer=l2(0.0001)))
            model.add(Activation('relu'))
            model.add(Conv2D(filters, kernel_size=(3, 3), padding='same',
                             kernel_initializer="he_uniform",
                             data_format='channels_last'))
            model.add(Dropout(0.2))

    def transition_down(self, filters):

        model = self.model
        model.add(BatchNormalization(mode=0, axis=1,
                                     gamma_regularizer=l2(0.0001),
                                     beta_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(Conv2D(filters, kernel_size=(1, 1), padding='same',
                         kernel_initializer="he_uniform"))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               data_format='channels_last'))

    def transition_up(self, filters, input_shape, output_shape):

        model = self.model
        model.add(Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2),
                                  padding='same',
                                  output_shape=output_shape,
                                  input_shape=input_shape,
                                  kernel_initializer="he_uniform",
                                  data_format='channels_last'))

    def create(self):

        model = self.model = models.Sequential()
        # cropping
        # model.add(Cropping2D(cropping=((68, 68), (128, 128)), input_shape=(3, 360,480)))

        model.add(Conv2D(48, kernel_size=(3, 3), padding='same',
                         input_shape=(224, 224, 3),
                         kernel_initializer="he_uniform",
                         kernel_regularizer=l2(0.0001),
                         data_format='channels_last'))

        # (5 * 4)* 2 + 5 + 5 + 1 + 1 +1
        # growth_m = 4 * 12
        # previous_m = 48

        self.dense_block(4, 112)  # 4*16 = 64 + 48 = 112
        self.transition_down(112)
        self.dense_block(5, 192)  # 5*16 = 80 + 112 = 192
        self.transition_down(192)
        self.dense_block(7, 304)  # 7 * 16 = 112 + 192 = 304
        self.transition_down(304)
        self.dense_block(10, 464)
        self.transition_down(464)
        self.dense_block(12, 656)
        self.transition_down(656)

        self.dense_block(15, 896)  # m = 656 + 15x16 = 896

        # upsampling part, m[B] is the sum of 3 terms
        # 1. the m value corresponding to same resolution in the downsampling part (skip connection)
        # 2. the number of feature maps from the upsampled block (n_layers[B-1] * growth_rate)
        # 3. the number of feature maps in the new block (n_layers[B] * growth_rate)
        #
        self.transition_up(1088, (1088, 7, 7), (None, 1088, 14, 14))  # m = 656 + 15x16 + 12x16 = 1088.
        self.dense_block(12, 1088)

        self.transition_up(816, (816, 14, 14), (None, 816, 28, 28))  # m = 464 + 12x16 + 10x16 = 816
        self.dense_block(10, 816)

        self.transition_up(576, (576, 28, 28), (None, 576, 56, 56))  # m = 304 + 10x16 + 7x16 = 576
        self.dense_block(7, 576)

        self.transition_up(384, (384, 56, 56), (None, 384, 112, 112))  # m = 192 + 7x16 + 5x16 = 384
        self.dense_block(5, 384)

        self.transition_up(256, (256, 112, 112), (None, 256, 224, 224))  # m = 112 + 5x16 + 4x16 = 256
        self.dense_block(4, 256)

        model.add(Conv2D(12, kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer="he_uniform",
                         kernel_regularizer=l2(0.0001),
                         data_format='channels_last'))

        model.add(Reshape((12, 224 * 224)))
        model.add(Permute((2, 1)))
        model.add(Activation('softmax'))
        model.summary()

        # TODO: Find a proper way of doing this
        #with open('tiramisu_fc_dense103_model.json', 'w') as outfile:
        #    outfile.write(json.dumps(json.loads(model.to_json()), indent=3))
