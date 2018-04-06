from typing import Tuple

from keras import Model, Input
from keras import backend as keras_backend
from keras.layers import MaxPooling2D, Conv2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras_contrib.losses import jaccard_distance

from utils.metrics import dice_coefficient


def unet(input_size: int, num_classes: int, channels: int = 3) -> Tuple[Model, str]:
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation

    https://arxiv.org/abs/1505.04597
    https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    """
    inputs = Input((input_size, input_size, channels), name="Input")
    with keras_backend.name_scope('Encode_1'):
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    with keras_backend.name_scope('Encode_2'):
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    with keras_backend.name_scope('Encode_3'):
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    with keras_backend.name_scope('Encode_4'):
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    with keras_backend.name_scope('Encode_5'):
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    with keras_backend.name_scope('Decode_1'):
        up1 = UpSampling2D(size=(2, 2), name='Upsampling_1')(conv5)
        merge6 = concatenate([up1, conv4], axis=3, name='Merge_1')
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    with keras_backend.name_scope('Decode_2'):
        up2 = UpSampling2D(size=(2, 2), name='Upsampling_2')(conv6)
        merge7 = concatenate([up2, conv3], axis=3, name='Merge_2')
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    with keras_backend.name_scope('Decode_3'):
        up3 = UpSampling2D(size=(2, 2), name='Upsampling_3')(conv7)
        merge8 = concatenate([up3, conv2], axis=3, name='Merge_3')
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    with keras_backend.name_scope('Decode_4'):
        up4 = UpSampling2D(size=(2, 2), name='Upsampling_4')(conv8)
        merge9 = concatenate([up4, conv1], axis=3, name='Merge_4')
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid', name='Classification')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4),
                  loss=jaccard_distance,
                  metrics=[dice_coefficient, jaccard_distance, 'accuracy'])

    return model, "unet"
