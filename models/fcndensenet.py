from typing import Tuple

from keras.models import Model
from keras.optimizers import Adam
from keras_contrib.applications.densenet import DenseNetFCN
from keras_contrib.losses import jaccard_distance

from utils.metrics import dice_coefficient


def fcndensenet(input_size: int, num_classes: int, loss, channels: int = 3) -> Tuple[Model, str]:
    model = DenseNetFCN(input_shape=(input_size, input_size, channels), classes=num_classes)
    model_name = 'fcn_densenet'

    model.compile(optimizer=Adam(), loss=loss, metrics=[dice_coefficient, jaccard_distance, 'accuracy'])

    return model, model_name
