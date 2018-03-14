import datetime
import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from keras.utils import plot_model
from keras_contrib.applications.densenet import DenseNetFCN

from models import fcn, unet, tiramisu, pspnet
from data_loader import Generator


def get_model(algorithm: str, input_size: int, num_classes: int):
    if algorithm is "fcn":
        model, model_name = fcn.fcn32(input_size=input_size, num_classes=num_classes)
    elif algorithm is "densefcn":
        model = DenseNetFCN(input_shape=(input_size, input_size, 3), classes=10)
        model_name = "fcn_densenet"
        from models.fcn import binary_crossentropy_with_logits
        model.compile(optimizer=SGD(lr=0.01 * (float(100) / 16), momentum=0.9),
                      loss=binary_crossentropy_with_logits,
                      metrics=['accuracy'])
    elif algorithm is "tiramisu":
        model, model_name = tiramisu.tiramisu(input_size=input_size, num_classes=num_classes)
    elif algorithm is "unet":
        model, model_name = unet.unet(input_size=input_size, num_classes=num_classes)
    elif algorithm is "pspnet":
        model, model_name = pspnet.pspnet(input_size=input_size, num_classes=num_classes)
    else:
        raise Exception("Invalid algorithm")

    return model, model_name


if __name__ == "__main__":
    # TODO: Load arguments from command line
    num_classes = 10
    input_size = 473
    epochs = 1000
    batch_size = 50
    val_amount = 20
    algorithm = "pspnet"

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
                             write_graph=False,
                             write_images=False)

    val_x, val_y = generator.next(amount=val_amount)

    print("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=batch_size, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, tbCallBack], validation_data=(val_x, val_y))
