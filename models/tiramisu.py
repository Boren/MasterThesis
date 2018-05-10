from typing import Tuple

from keras import Input, Model
from keras import backend as keras_backend
from keras.layers import Conv2D, Conv2DTranspose, concatenate
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras_contrib.losses import jaccard_distance

from utils.metrics import dice_coefficient


def tiramisu(input_size: int, num_classes: int, loss, channels: int = 3) -> Tuple[Model, str]:
    """
    The One Hundred Layers Tiramisu:
    Fully Convolutional DenseNets for Semantic Segmentation

    https://arxiv.org/abs/1611.09326
    https://github.com/0bserver07/One-Hundred-Layers-Tiramisu
    https://github.com/SimJeg/FC-DenseNet

    103 layers - High memory requirements
    """
    n_filters_first_conv = 48
    n_pool = 4
    growth_rate = 12
    n_layers_per_block = 5
    n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    dropout_p = 0.2

    inputs = Input((input_size, input_size, channels), name="Input")

    stack = Conv2D(n_filters_first_conv, kernel_size=(3, 3), padding='same')(inputs)
    n_filters = n_filters_first_conv

    # Start downsampling
    skip_connection_list = []

    for i in range(n_pool):
        # Dense Block
        with keras_backend.name_scope("Dense_Block"):
            for j in range(n_layers_per_block[i]):
                x = dense_block(stack, growth_rate, dropout_p=dropout_p)
                stack = concatenate([stack, x])
                n_filters += growth_rate

            skip_connection_list.append(stack)

        # Transition Down
        with keras_backend.name_scope("Transition_Down"):
            stack = transition_down(stack, n_filters, dropout_p)

    skip_connection_list = skip_connection_list[::-1]

    # Bottleneck
    block_to_upsample = []

    # Dense Block
    with keras_backend.name_scope("Dense_Block"):
        for j in range(n_layers_per_block[n_pool]):
            x = dense_block(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(x)
            stack = concatenate([stack, x])

    # Begin upsampling
    for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = transition_up(skip_connection_list[i], block_to_upsample, n_filters_keep)

        # Dense Block
        block_to_upsample = []
        with keras_backend.name_scope("Dense_Block"):
            for j in range(n_layers_per_block[n_pool + i + 1]):
                x = dense_block(stack, growth_rate, dropout_p=dropout_p)
                block_to_upsample.append(x)
                stack = concatenate([stack, x])

    # Classifier
    with keras_backend.name_scope("Classification"):
        x = Conv2D(num_classes, kernel_size=(1, 1), padding='same')(stack)
        x = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(), loss=loss, metrics=[dice_coefficient, jaccard_distance, 'accuracy'])

    return model, "tiramisu"


def dense_block(x, filters, filter_size=3, dropout_p=0.2):
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size=(filter_size, filter_size), padding='same', activation='relu')(x)
    x = Dropout(dropout_p)(x)

    return x


def transition_down(x, n_filters, dropout_p=0.2):
    with keras_backend.name_scope("Transition_Down"):
        x = dense_block(x, n_filters, filter_size=1, dropout_p=dropout_p)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        return x


def transition_up(skip_connection, block_to_upsample, n_filters_keep):
    with keras_backend.name_scope("Transition_Up"):
        x = concatenate(block_to_upsample)
        x = Conv2DTranspose(n_filters_keep, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = concatenate([x, skip_connection])

        return x
