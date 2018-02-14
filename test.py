import os

import numpy as np
import matplotlib.pyplot as plt
from utils.visualize import CLASS_TO_LABEL, mask_for_array

from models import unet, fcn
from data_loader import Generator

if __name__ == "__main__":
    num_classes = 10
    input_size = 160

    generator = Generator(patch_size=input_size)
    model = unet.unet(input_size=input_size, num_classes=num_classes)

    # TODO: Dynamic weight loading from argument
    if not os.path.isfile('weights/unet.hdf5'):
        print("No weights to load")

    print("Loading saved weights")
    model.load_weights('weights/unet.hdf5')

    test_amount = 1

    #test_x, test_y = generator.next(amount=test_amount)
    #test_y_result = model.predict_generator(generator.generator(), steps=test_amount, verbose=1)

    # OK test area with buildings, roads etc.
    test_x_temp, test_y_temp = generator.get_patch(image="6060_2_3",
                                                   x=811, y=1512,
                                                   width=160, height=160)

    test_x = np.array([test_x_temp])
    test_y = np.array([test_y_temp])

    test_y_result = model.predict(test_x, batch_size=1, verbose=1)

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
            ax2.imshow(test_y[patchnum, :, :, cls], cmap=plt.get_cmap('gray'))

            ax3 = plt.subplot(11, 2, 2 * cls + 4)
            ax3.set_title('Prediction ({cls})'.format(cls=CLASS_TO_LABEL[cls + 1]))
            ax3.imshow(test_y_result[patchnum, :, :, cls], cmap=plt.get_cmap('gray'),
                       interpolation='nearest', vmin=0, vmax=1)
        plt.show()
