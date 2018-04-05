import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import webcolors
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model

from data_loader import Generator
from models import fcndensenet, unet, tiramisu, pspnet
from utils.visualize import COLOR_MAPPING, CLASS_TO_LABEL, mask_for_array


def get_model(algorithm: str, input_size: int, num_classes: int):
    if algorithm == 'fcn_densenet':
        model, model_name = fcndensenet.fcndensenet(input_size=input_size, num_classes=num_classes)
    elif algorithm == 'tiramisu':
        model, model_name = tiramisu.tiramisu(input_size=input_size, num_classes=num_classes)
    elif algorithm == 'unet':
        model, model_name = unet.unet(input_size=input_size, num_classes=num_classes)
    elif algorithm == 'pspnet':
        model, model_name = pspnet.pspnet(input_size=input_size, num_classes=num_classes)
    else:
        raise Exception('{} is an invalid algorithm'.format(algorithm))

    return model, model_name


def train(algorithm: str, input_size: int, epochs: int, batch_size: int, num_classes: int = 10):
    val_amount = max(batch_size // 10, 1)

    generator = Generator(patch_size=input_size,
                          batch_size=batch_size)

    model, model_name = get_model(algorithm, input_size, num_classes)

    if os.path.isfile('weights/{}.hdf5'.format(model_name)):
        load = input("Saved weights found. Load? (y/n)")
        if load == "y":
            print("Loading saved weights")
            model.load_weights('weights/{}.hdf5'.format(model_name))
    model.summary()

    if not os.path.exists('images'):
        os.makedirs('images')
    plot_model(model, show_shapes=False, to_file="images/{}_model.png".format(model_name))
    plot_model(model, show_shapes=True, to_file="images/{}_model_shapes.png".format(model_name))

    if not os.path.exists('weights'):
        os.makedirs('weights')
    model_checkpoint = ModelCheckpoint('weights/{}.hdf5'.format(model_name), monitor='loss', save_best_only=True)

    # Setup tensorboard model
    timenow = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    tensorboard_callback = TensorBoard(log_dir='tensorboard_log/{}_{}/'.format(model_name, timenow),
                                       histogram_freq=0,
                                       write_graph=True,
                                       write_images=False)

    val_x, val_y = generator.next(amount=val_amount)

    print("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=batch_size, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, tensorboard_callback], validation_data=(val_x, val_y))


def test(algorithm: str, input_size: int, num_classes: int = 10):
    test_image = '6060_2_3'
    generator = Generator(patch_size=input_size)

    model, model_name = get_model(algorithm, input_size, num_classes)

    if os.path.isfile('weights/{}.hdf5'.format(model_name)):
        print("Loading saved weights from weights/{}.hdf5".format(model_name))
        model.load_weights('weights/{}.hdf5'.format(model_name))
    else:
        raise Exception("No weights")

    test_amount = 1

    # test_x, test_y = generator.next(amount=test_amount)
    # test_y_result = model.predict_generator(generator.generator(), steps=test_amount, verbose=1)

    # OK test area with buildings, roads etc.
    test_x_temp, test_y_temp = generator.get_patch(image=test_image,
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
    result_img.save(os.path.join("images", '{}_combined.png'.format(test_image)))

    # Plot results
    print("Plotting results...")
    for patchnum in range(test_amount):
        plt.figure(figsize=(2000 / 96, 5000 / 96), dpi=96)
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
        plt.savefig(os.path.join("images", '{}.png'.format(test_image)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", help="Which algorithm to train/test")
    parser.add_argument("--size", help="Size of image patches to train/test on", default=160, type=int)
    parser.add_argument("--epochs", help="How many epochs to run", default=1000, type=int)
    parser.add_argument("--batch", help="How many samples in a batch", default=100, type=int)
    parser.add_argument("--test", help="Run a test", dest='test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    algorithm = args.algorithm
    input_size = args.size
    epochs = args.epochs
    batch_size = args.batch

    num_classes = 10

    if args.test:
        test(algorithm, input_size, num_classes)
    else:
        train(algorithm, input_size, epochs, batch_size, num_classes)


if __name__ == "__main__":
    main()
