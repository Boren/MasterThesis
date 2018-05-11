import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import webcolors
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from termcolor import colored

from data_loader import Generator
from models import fcndensenet, unet, tiramisu, pspnet
from utils.visualize import COLOR_MAPPING, CLASS_TO_LABEL
from utils.loss import jaccard_loss, dice_loss, ce_dice_loss, ce_jaccard_loss

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler('run.log')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


def get_model(algorithm: str, input_size: int, num_classes: int, loss, channels: int = 3, ):
    if algorithm == 'fcn_densenet':
        model, model_name = fcndensenet.fcndensenet(input_size=input_size, num_classes=num_classes, loss=loss, channels=channels)
    elif algorithm == 'tiramisu':
        model, model_name = tiramisu.tiramisu(input_size=input_size, num_classes=num_classes, loss=loss, channels=channels)
    elif algorithm == 'unet':
        model, model_name = unet.unet(input_size=input_size, num_classes=num_classes, loss=loss, channels=channels)
    elif algorithm == 'pspnet':
        model, model_name = pspnet.pspnet(input_size=input_size, num_classes=num_classes, loss=loss, channels=channels)
    else:
        raise Exception('{} is an invalid algorithm'.format(algorithm))

    return model, model_name


def get_loss(loss: str = 'crossentropy'):
    if loss == 'crossentropy':
        return binary_crossentropy
    elif loss == 'jaccard':
        return jaccard_loss
    elif loss == 'dice':
        return dice_loss
    elif loss == 'cejaccard':
        return ce_jaccard_loss
    elif loss == 'cedice':
        return ce_dice_loss


def create_directories(run_name: str):
    os.makedirs('images/{}'.format(run_name), exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('tensorboard_log', exist_ok=True)


def train(args):
    generator = Generator(patch_size=args.size, batch_size=args.batch, channels=args.channels, augment=args.augmentation)

    model, model_name = get_model(args.algorithm, args.size, args.classes, get_loss(args.loss), args.channels)

    if args.name:
        run_name = "{}_{}".format(model_name, args.name)
    else:
        timenow = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        run_name = "{}_{}_{}channel_{}".format(model_name, timenow, args.channels, args.loss)

    create_directories(run_name)

    # TODO: Update with ability to choose weights
    if os.path.isfile('weights/{}.hdf5'.format(model_name)):
        load = input("Saved weights found. Load? (y/n)")
        if load == "y":
            print("Loading saved weights")
            model.load_weights('weights/{}.hdf5'.format(model_name))

    if args.verbose:
        model.summary()

    # Some doesn't have graphviz installed. Skip if not installed.
    try:
        plot_model(model, os.path.join('images', run_name, 'model.png'))
        plot_model(model, os.path.join('images', run_name, 'model_shapes.png'), show_shapes=True)
    except Exception as e:
        logger.warning("GraphViz missing. Skipping model plot")
        logger.warning(e)

    model_checkpoint = ModelCheckpoint('weights/{}.hdf5'.format(run_name), monitor='val_loss', save_best_only=True)

    # Setup tensorboard model
    tensorboard_callback = TensorBoard(log_dir='tensorboard_log/{}/'.format(run_name), histogram_freq=0, write_graph=True, write_images=False)

    val_x, val_y = generator.get_validation_data()

    logger.debug("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=args.batch,
                        epochs=args.epochs, verbose=1 if args.verbose else 2,
                        callbacks=[model_checkpoint, tensorboard_callback],
                        validation_data=(val_x, val_y))


def select_weights(model_name: str):
    weight_files = [filename for filename in os.listdir('weights') if filename.startswith(model_name)]

    if len(weight_files) > 0:
        if len(weight_files) == 1:
            selected = 0
        else:
            for i, weight in enumerate(weight_files):
                print('{}:  {}'.format(i, weight))
            selected = int(input("Select a weight file: "))
        return weight_files[selected]
    else:
        raise Exception("No weights for {}".format(model_name))


def calculate_mean_iou(y_true, y_pred, num_classes):
    mean_iou = []

    for cls in range(num_classes):
        cls = cls + 1

        print('Calculating IoU for {}'.format(CLASS_TO_LABEL[cls]))

        y_true_cls = (y_true == cls).astype(int)
        y_pred_cls = (y_pred == cls).astype(int)

        true_positive = np.sum(np.logical_and(y_pred_cls == 1, y_true_cls == 1))
        print("- True Positive {}".format(true_positive))

        true_negative = np.sum(np.logical_and(y_pred_cls == 0, y_true_cls == 0))
        print("- True Negative {}".format(true_negative))

        false_positive = np.sum(np.logical_and(y_pred_cls == 1, y_true_cls == 0))
        print("- False Positive {}".format(false_positive))

        false_negative = np.sum(np.logical_and(y_pred_cls == 0, y_true_cls == 1))
        print("- False Negative {}".format(false_negative))

        score = true_positive / (false_positive + false_negative + true_positive + 0.0001)
        print('- IoU: {}'.format(score))

        mean_iou.append(score)

    return np.mean(mean_iou)


# TODO: Fix
def print_confusion_matrix(y_true, y_pred, num_classes):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    cnf_text = np.array([[x if x < 10000 else "" for x in l] for l in cnf_matrix])

    df_cm = pd.DataFrame(cnf_matrix, index=[i for i in ["BG"] + list(CLASS_TO_LABEL.values())],
                         columns=[i for i in ["BG"] + list(CLASS_TO_LABEL.values())])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=cnf_text, fmt="s")
    #plt.savefig(os.path.join(save_folder, '{}_confusion_matrix.png'.format(test_image)))


def test(args):
    prediction_cutoff = 0.5
    generator = Generator(patch_size=args.size, channels=args.channels)

    model, model_name = get_model(args.algorithm, args.size, args.classes, get_loss(args.loss), args.channels)

    weight_file = select_weights(model_name)
    logger.debug("Loading saved weights from weights/{}".format(weight_file))
    model.load_weights('weights/{}'.format(weight_file))

    create_directories(os.path.splitext(weight_file)[0])
    save_folder = os.path.join('images', os.path.splitext(weight_file)[0])

    test_images = ['6140_3_1', '6180_4_3']

    for test_image in test_images:
        logger.debug('Testing image {}'.format(test_image))
        test_x, test_y, new_size, splits, w, h = generator.get_test_patches(image=test_image, network_size=args.size)

        cutoff_array = np.full((len(test_x), args.size, args.size, 1), fill_value=prediction_cutoff)

        test_y_result = model.predict(test_x, batch_size=1, verbose=1)
        test_y_result = np.append(cutoff_array, test_y_result, axis=3)

        out = np.zeros((new_size, new_size, args.classes + 1))

        for row in range(splits):
            for col in range(splits):
                out[args.size * row:args.size * (row + 1), args.size * col:args.size * (col + 1), :] = test_y_result[row * splits + col, :, :, :]

        result = np.argmax(np.squeeze(out), axis=-1).astype(np.uint8)
        result = result[:w, :h]

        palette = []

        palette.extend([255, 255, 255])

        for i in range(args.classes):
            palette.extend(list(webcolors.hex_to_rgb(COLOR_MAPPING[int(i + 1)])))

        # for i in range(len(test_x)):
        result_img = Image.fromarray(result, mode='P')
        result_img.putpalette(palette)
        result_img.save(os.path.join(save_folder, '{}_{}.png'.format(test_image, model_name)))

        if test_y is not None:
            y_train = np.load(os.path.join('data/cache/{}_y.npy'.format(test_image)), mmap_mode='r')
            y_mask = generator.flatten(y_train)
            result_img = Image.fromarray(y_mask, mode='P')
            result_img.putpalette(palette)
            result_img.save(os.path.join(save_folder, '{}_gt.png'.format(test_image)))

            y_mask_flat = y_mask.flatten()
            result_flat = result.flatten()

            mean_iou = calculate_mean_iou(y_mask_flat, result_flat, args.classes)

            print('Mean IoU: {}'.format(mean_iou))

            # print_confusion_matrix(y_mask_flat, result_flat, len(args.classes))

    # Old plotting methods.
    # Maybe we need some of this later
    '''
    print("Plotting results...")
    for patchnum in range(test_amount):
        plt.figure(figsize=(2000 / 96, 5000 / 96), dpi=96)
        ax1 = plt.subplot(11, 3, 1)
        ax1.set_title('Raw RGB data')
        ax1.imshow(test_x[patchnum, :, :, :], cmap=plt.get_cmap('gist_ncar'))

        ax1 = plt.subplot(11, 3, 2)
        ax1.set_title('Combined ground truth')
        ax1.imshow(mask_for_array(test_y[patchnum, :, :, :]), cmap=plt.get_cmap('gist_ncar'))

        ax1 = plt.subplot(11, 3, 3)
        ax1.set_title('Combined predicition')
        ax1.imshow(result_img)

        for cls in range(num_classes):
            ax2 = plt.subplot(11, 3, 3 * cls + 4)
            ax2.set_title('Ground Truth ({cls})'.format(cls=CLASS_TO_LABEL[cls + 1]))
            ax2.imshow(test_y[patchnum, :, :, cls], cmap=plt.get_cmap('Reds'))

            ax3 = plt.subplot(11, 3, 3 * cls + 5)
            ax3.set_title('Prediction ({cls})'.format(cls=CLASS_TO_LABEL[cls + 1]))
            ax3.imshow(test_y_result[patchnum, :, :, cls], cmap=plt.get_cmap('Reds'),
                       interpolation='nearest')

            ax4 = plt.subplot(11, 3, 3 * cls + 6)
            ax4.set_title('Prediction ({cls})'.format(cls=CLASS_TO_LABEL[cls + 1]))
            ax4.imshow(test_y_result[patchnum, :, :, cls], cmap=plt.get_cmap('Reds'),
                       interpolation='nearest', vmin=0, vmax=1)
        plt.suptitle('{}'.format(algorithm))
        plt.savefig(os.path.join(save_folder, '{}.png'.format(test_image)))
    '''


def print_options(args):
    if args.test:
        run_type = colored('TEST', 'red')
    else:
        run_type = colored('TRAINING', 'red')

    print()
    print("Starting {} run with following options:".format(run_type))
    print("- Algorithm: {}".format(colored(args.algorithm, 'green')))
    print("- Patch size: {}".format(colored(args.size, 'green')))
    print("- Channels: {}".format(colored(args.channels, 'green')))

    if args.loss == "crossentropy":
        loss = "Crossentropy"
    elif args.loss == "jaccard":
        loss = "Jaccard"
    elif args.loss == "dice":
        loss = "Dice"
    elif args.loss == "cejaccard":
        loss = "Crossentropy + Jaccard"
    elif args.loss == "cedice":
        loss = "Crossentropy + Dice"
    else:
        raise Exception("Invalid loss function")

    print('- Loss function: {}'.format(colored(loss, 'green')))
    if not args.test:
        print("- Epochs: {}".format(colored(args.epochs, 'green')))
        print("- Batch size: {}".format(colored(args.batch, 'green')))
        if args.augmentation:
            print("- Augmentation: {}".format(colored('ON', 'green')))
        else:
            print("- Augmentation: {}".format(colored('OFF', 'red')))

    classes = '\n' + '\n'.join(["    " + CLASS_TO_LABEL[x + 1] for x in range(args.classes)])

    print("- Classes: {}".format(colored(classes, 'green')))
    print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', help='Which algorithm to train/test', choices=['unet', 'fcn_densenet', 'tiramisu', 'pspnet'])
    parser.add_argument('--size', default=320, type=int, help='Size of image patches to train/test on')
    parser.add_argument('--epochs', default=1000, type=int, help='How many epochs to run')
    parser.add_argument('--batch', default=100, type=int, help='How many samples in a batch')
    parser.add_argument('--channels', default=3, type=int, help='How many channels.', choices=[3, 8, 16])
    parser.add_argument('--loss', default='crossentropy', type=str, help='Which loss function to use.', choices=['crossentropy', 'jaccard', 'dice', 'cejaccard', 'cedice'])
    parser.add_argument('--test', dest='test', action='store_true', help='Run a test')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Show additional debug information')
    parser.add_argument('--noaugment', dest='noaugment', action='store_true', help='Disable data augmenation')
    parser.add_argument('--name', default=None, type=str, help='Give the run a name')

    parser.set_defaults(test=False, verbose=False, noaugment=False)

    args = parser.parse_args()
    args.classes = 8
    args.augmentation = not args.noaugment

    print_options(args)

    if args.test:
        test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
