import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import webcolors
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard

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


def create_directories():
    if not os.path.exists('images'):
        os.makedirs('images')

    if not os.path.exists('weights'):
        os.makedirs('weights')

    if not os.path.exists('tensorboard_log'):
        os.makedirs('tensorboard_log')


def train(algorithm: str, input_size: int, epochs: int,
          batch_size: int, num_classes: int = 10):
    create_directories()

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

    timenow = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    model_checkpoint = ModelCheckpoint('weights/{}_{}.hdf5'.format(model_name, timenow), monitor='val_loss', save_best_only=True)

    # Setup tensorboard model
    tensorboard_callback = TensorBoard(log_dir='tensorboard_log/{}_{}/'.format(model_name, timenow),
                                       histogram_freq=0,
                                       write_graph=True,
                                       write_images=False)

    val_x, val_y = generator.next(amount=val_amount, data_type='validation')

    print("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=batch_size, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, tensorboard_callback], validation_data=(val_x, val_y))


def test(algorithm: str, input_size: int, num_classes: int = 10):
    test_image = '6060_2_3'
    generator = Generator(patch_size=input_size)

    model, model_name = get_model(algorithm, input_size, num_classes)

    weight_files = [filename for filename in os.listdir('weights') if filename.startswith(algorithm)]
    if len(weight_files) > 0:
        for i, weight in enumerate(weight_files):
            print('{}:  {}'.format(i, weight))

        selected = int(input("Select a weight file: "))
        selected_weight = weight_files[selected]
        print("Loading saved weights from weights/{}".format(selected_weight))
        model.load_weights('weights/{}'.format(selected_weight))
    else:
        raise Exception("No weights")

    test_amount = 1

    # OK test area with buildings, roads etc.
    test_x_temp, test_y_temp = \
        generator.get_patch(image=test_image,
                            x=811, y=1512,
                            width=input_size, height=input_size)

    test_x = np.array([test_x_temp])
    test_y = np.array([test_y_temp])

    test_y_result = model.predict(test_x, batch_size=1, verbose=1)

    result = np.argmax(np.squeeze(test_y_result), axis=-1).astype(np.uint8)
    result_img = Image.fromarray(result, mode='P')

    palette = []

    for i in range(num_classes):
        palette.extend(list(webcolors.hex_to_rgb(COLOR_MAPPING[int(i+1)])))

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
        #ax1.imshow(mask_for_array(test_y[patchnum, :, :, :]), cmap=plt.get_cmap('gist_ncar'))
        ax1.imshow(result_img)

        for cls in range(num_classes):
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

    parser.add_argument("--algorithm",
                        help="Which algorithm to train/test")

    parser.add_argument("--size", default=160, type=int,
                        help="Size of image patches to train/test on")

    parser.add_argument("--epochs", default=1000, type=int,
                        help="How many epochs to run")

    parser.add_argument("--batch", default=100, type=int,
                        help="How many samples in a batch")

    parser.add_argument("--test", dest='test', action='store_true',
                        help="Run a test")

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
