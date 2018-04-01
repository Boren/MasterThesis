import os

import numpy as np
import matplotlib.pyplot as plt
import webcolors
from PIL import Image

import tensorflow as tf
from keras import backend as keras_backend
from tensorflow.python import debug as tf_debug

from utils.visualize import COLOR_MAPPING, CLASS_TO_LABEL, mask_for_array

from data_loader import Generator

from train import get_model

if __name__ == "__main__":
    #keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "Chronos-Manjaro:6064"))

    num_classes = 10
    input_size = 320
    algorithm = "fcn"

    generator = Generator(patch_size=input_size)

    model, model_name = get_model(algorithm, input_size, num_classes)

    if os.path.isfile('weights/{}.hdf5'.format(model_name)):
        print("Loading saved weights from weights/{}.hdf5".format(model_name))
        model.load_weights('weights/{}.hdf5'.format(model_name))
    else:
        raise Exception("No weights")

    test_amount = 1

    #test_x, test_y = generator.next(amount=test_amount)
    #test_y_result = model.predict_generator(generator.generator(), steps=test_amount, verbose=1)

    # OK test area with buildings, roads etc.
    test_x_temp, test_y_temp = generator.get_patch(image="6060_2_3",
                                                   x=811, y=1512,
                                                   width=input_size, height=input_size)

    test_x = np.array([test_x_temp])
    test_y = np.array([test_y_temp])

    test_y_result = model.predict(test_x, batch_size=1, verbose=1)

    result = np.argmax(np.squeeze(test_y_result), axis=-1).astype(np.uint8)
    result_img = Image.fromarray(result, mode='P')

    palette = []

    for i in range(1, 11):
        palette.extend(list(webcolors.hex_to_rgb(COLOR_MAPPING[int(i)])))

    result_img.putpalette(palette)
    result_img.save(os.path.join("images", '6060_2_3.png'))

    # Plot results
    print("Plotting results...")
    for patchnum in range(test_amount):
        fig = plt.figure(figsize=(2000/96, 5000/96), dpi=96)
        ax1 = plt.subplot(11, 2, 1)
        ax1.set_title('Raw RGB data')
        ax1.imshow(test_x[patchnum, :, :, :], cmap=plt.get_cmap('gist_ncar'))

        ax1 = plt.subplot(11, 2, 2)
        ax1.set_title('Combined ground truth')
        ax1.imshow(mask_for_array(test_y[patchnum, :, :, :]), cmap=plt.get_cmap('gist_ncar'))

        for cls in range(10):
            ax2 = plt.subplot(11, 2, 2 * cls + 3)
            ax2.set_title('Ground Truth ({cls})'.format(cls=CLASS_TO_LABEL[cls + 1]))
            ax2.imshow(test_y[patchnum, :, :, cls], cmap=plt.get_cmap('Reds'))

            ax3 = plt.subplot(11, 2, 2 * cls + 4)
            ax3.set_title('Prediction ({cls})'.format(cls=CLASS_TO_LABEL[cls + 1]))
            ax3.imshow(test_y_result[patchnum, :, :, cls], cmap=plt.get_cmap('Reds'),
                       interpolation='nearest')
        plt.suptitle('{}'.format(algorithm))
        plt.show()
