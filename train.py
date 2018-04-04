import argparse
import datetime
import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from keras.utils import plot_model
from keras_contrib.applications.densenet import DenseNetFCN

from models import fcn, unet, tiramisu, pspnet
from data_loader import Generator


def get_model(algorithm: str, input_size: int, num_classes: int):
    if algorithm == 'fcn':
        model, model_name = fcn.fcn32(input_size=input_size, num_classes=num_classes)
    elif algorithm == 'densefcn':
        model = DenseNetFCN(input_shape=(input_size, input_size, 3), classes=10)
        model_name = 'fcn_densenet'
        from models.fcn import binary_crossentropy_with_logits
        model.compile(optimizer=SGD(lr=0.01 * (float(100) / 16), momentum=0.9),
                      loss=binary_crossentropy_with_logits,
                      metrics=['accuracy'])
    elif algorithm == 'tiramisu':
        model, model_name = tiramisu.tiramisu(input_size=input_size, num_classes=num_classes)
    elif algorithm == 'unet':
        model, model_name = unet.unet(input_size=input_size, num_classes=num_classes)
    elif algorithm == 'pspnet':
        model, model_name = pspnet.pspnet(input_size=input_size, num_classes=num_classes)
    else:
        raise Exception('{} is an invalid algorithm'.format(algorithm))

    return model, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", help="Which algorithm to train/test")
    parser.add_argument("--epochs", help="How many epochs to run", default=1000, type=int)
    parser.add_argument("--size", help="Size of image patches to train/test on", default=160, type=int)
    parser.add_argument("--batch", help="How many samples in a batch", default=100, type=int)
    args = parser.parse_args()

    num_classes = 10
    input_size = args.size
    epochs = args.epochs
    batch_size = args.batch
    val_amount = batch_size // 10
    algorithm = args.algorithm

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
    tbCallBack = TensorBoard(log_dir='tensorboard_log/{}_{}/'.format(model_name, timenow),
                             histogram_freq=0,
                             write_graph=True,
                             write_images=False)

    val_x, val_y = generator.next(amount=val_amount)

    print("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=batch_size, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, tbCallBack], validation_data=(val_x, val_y))
