import argparse
import datetime
import os

import numpy as np
import webcolors
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from sklearn.metrics import jaccard_similarity_score, confusion_matrix

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import Generator
from models import fcndensenet, unet, tiramisu, pspnet
from utils.visualize import COLOR_MAPPING, CLASS_TO_LABEL


def get_model(algorithm: str, input_size: int, num_classes: int):
    if algorithm == 'fcn_densenet':
        model, model_name = \
            fcndensenet.fcndensenet(input_size=input_size,
                                    num_classes=num_classes)
    elif algorithm == 'tiramisu':
        model, model_name = \
            tiramisu.tiramisu(input_size=input_size, num_classes=num_classes)
    elif algorithm == 'unet':
        model, model_name = \
            unet.unet(input_size=input_size, num_classes=num_classes)
    elif algorithm == 'pspnet':
        model, model_name = \
            pspnet.pspnet(input_size=input_size, num_classes=num_classes)
    else:
        raise Exception('{} is an invalid algorithm'.format(algorithm))

    return model, model_name


def create_directories(run_name: str):
    os.makedirs('images', exist_ok=True)
    os.makedirs('images/{}'.format(run_name), exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('tensorboard_log', exist_ok=True)


def train(algorithm: str, input_size: int, epochs: int, batch_size: int,
          num_classes: int = 10, verbose: bool = False):
    val_amount = max(batch_size // 10, 1)

    generator = Generator(patch_size=input_size,
                          batch_size=batch_size)

    model, model_name = get_model(algorithm, input_size, num_classes)

    timenow = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_name = "{}_{}".format(model_name, timenow)
    create_directories(run_name)

    # TODO: Update with ability to choose weights
    if os.path.isfile('weights/{}.hdf5'.format(model_name)):
        load = input("Saved weights found. Load? (y/n)")
        if load == "y":
            print("Loading saved weights")
            model.load_weights('weights/{}.hdf5'.format(model_name))
    if verbose:
        model.summary()

    # Some doesn't have graphviz installed. Skip if not installed.
    try:
        plot_model(model, os.path.join('images', run_name, 'model.png'))
        plot_model(model, os.path.join('images', run_name, 'model_shapes.png'),
                   show_shapes=True)
    except:
        print("Skipping model plot")

    model_checkpoint = \
        ModelCheckpoint('weights/{}.hdf5'.format(run_name),
                        monitor='val_loss', save_best_only=True)

    # Setup tensorboard model
    tensorboard_callback = \
        TensorBoard(log_dir='tensorboard_log/{}/'.format(run_name),
                    histogram_freq=0, write_graph=True, write_images=False)

    val_x, val_y = generator.next(amount=val_amount, data_type='validation')

    print("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=batch_size,
                        epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, tensorboard_callback],
                        validation_data=(val_x, val_y))


def test(algorithm: str, input_size: int, num_classes: int = 10,
         verbose: bool = False, prediction_cutoff: float = 0.5):
    generator = Generator(patch_size=input_size)

    model, model_name = get_model(algorithm, input_size, num_classes)

    weight_files = [filename for filename in os.listdir('weights')
                    if filename.startswith(model_name)]

    if len(weight_files) > 0:
        if len(weight_files) == 1:
            selected = 0
        else:
            for i, weight in enumerate(weight_files):
                print('{}:  {}'.format(i, weight))
            selected = int(input("Select a weight file: "))
        selected_weight = weight_files[selected]
        print("Loading saved weights from weights/{}".format(selected_weight))
        model.load_weights('weights/{}'.format(selected_weight))
    else:
        raise Exception("No weights")

    create_directories(os.path.splitext(selected_weight)[0])
    save_folder = os.path.join('images', os.path.splitext(selected_weight)[0])

    test_images = ['6140_3_1', '6100_2_3', '6180_4_3']
    # test_images = [img for img in generator.all_image_ids if img not in generator.training_image_ids]

    for test_image in test_images:
        print('Testing image {}'.format(test_image))
        test_x, test_y, new_size, splits, w, h = \
            generator.get_test_patches(image=test_image,
                                       network_size=input_size)

        cutoff_array = np.full((len(test_x), input_size, input_size, 1),
                               fill_value=prediction_cutoff)

        test_y_result = model.predict(test_x, batch_size=1, verbose=1)
        test_y_result = np.append(cutoff_array, test_y_result, axis=3)

        out = np.zeros((new_size, new_size, num_classes + 1))

        for row in range(splits):
            for col in range(splits):
                out[input_size * row:input_size * (row + 1),
                input_size * col:input_size * (col + 1), :] = test_y_result[
                                                              row * splits + col,
                                                              :, :, :]

        result = np.argmax(np.squeeze(out), axis=-1).astype(np.uint8)
        result = result[:w, :h]

        palette = []

        palette.extend([255, 255, 255])

        for i in range(num_classes):
            palette.extend(
                list(webcolors.hex_to_rgb(COLOR_MAPPING[int(i + 1)])))

        # for i in range(len(test_x)):
        result_img = Image.fromarray(result, mode='P')
        result_img.putpalette(palette)
        result_img.save(
            os.path.join(save_folder, '{}_combined.png'.format(test_image)))

        if test_y is not None:
            y_train = np.load(os.path.join('data/cache/{}_y.npy'.format(test_image)))
            y_mask = generator.flatten(y_train)
            result_img = Image.fromarray(y_mask, mode='P')
            result_img.putpalette(palette)
            result_img.save(os.path.join(save_folder,
                                         '{}_gt.png'.format(test_image)))

            y_mask_flat = y_mask.flatten()
            result_flat = result.flatten()

            cnf_matrix = confusion_matrix(y_mask_flat, result_flat)
            cnf_text = np.array([[x if x < 1000 else "" for x in l] for l in cnf_matrix])

            df_cm = pd.DataFrame(cnf_matrix, index = [i for i in ["BG"] + list(CLASS_TO_LABEL.values())],
                                      columns = [i for i in ["BG"] + list(CLASS_TO_LABEL.values())])
            plt.figure(figsize = (10,7))
            sn.heatmap(df_cm, annot=cnf_text, fmt="s")
            plt.savefig("test.png")

            for cls in range(num_classes):
                cls = cls+1
                score = jaccard_similarity_score(
                    [p if p == cls else 0 for p in y_mask_flat],
                    [p if p == cls else 0 for p in result_flat])
                print('{}: {}'.format(CLASS_TO_LABEL[cls], score))
    # Plot results
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm",
                        help="Which algorithm to train/test")

    parser.add_argument("--size", default=256, type=int,
                        help="Size of image patches to train/test on")

    parser.add_argument("--epochs", default=1000, type=int,
                        help="How many epochs to run")

    parser.add_argument("--batch", default=100, type=int,
                        help="How many samples in a batch")

    parser.add_argument("--test", dest='test', action='store_true',
                        help="Run a test")

    parser.add_argument("--verbose", dest='verbose', action='store_true',
                        help="Show additional debug information")

    parser.set_defaults(test=False, verbose=False)
    args = parser.parse_args()

    algorithm = args.algorithm
    input_size = args.size
    epochs = args.epochs
    batch_size = args.batch
    verbose = args.verbose

    num_classes = 10

    if args.test:
        test(algorithm, input_size, num_classes, verbose)
    else:
        train(algorithm, input_size, epochs, batch_size, num_classes, verbose)


if __name__ == "__main__":
    main()
