import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model

from models import fcn, unet, tiramisu
from data_loader import Generator


if __name__ == "__main__":
    # TODO: Load arguments from command line
    num_classes = 10
    input_size = 160
    epochs = 1
    batch_size = 1
    val_amount = 1
    algorithm = "fcn"

    generator = Generator(patch_size=input_size, batch_size=batch_size)

    if algorithm is "fcn":
        model, model_name = fcn.fcn32(input_size=input_size, num_classes=num_classes)
    elif algorithm is "tiramisu":
        model, model_name = tiramisu.tiramisu(input_size=input_size, num_classes=num_classes)
    elif algorithm is "unet":
        model, model_name = unet.unet(input_size=input_size, num_classes=num_classes)
    else:
        raise Exception("Invalid algorithm")

    if os.path.isfile('weights/{}.hdf5'.format(model_name)):
        load = input("Saved weights found. Load? (y/n)")
        if load == "y":
            print("Loading saved weights")
            model.load_weights('weights/{}.hdf5'.format(model_name))
    model.summary()
    plot_model(model, show_shapes=False, to_file="images/{}_model.png".format(model_name))
    plot_model(model, show_shapes=True, to_file="images/{}_model_shapes.png".format(model_name))
    if not os.path.exists('weights'):
        os.makedirs('weights')
    model_checkpoint = ModelCheckpoint('weights/{}.hdf5'.format(model_name), monitor='loss', save_best_only=True)

    # Setup tensorboard model
    tbCallBack = TensorBoard(log_dir='tensorboard_log/{}/'.format(model_name),
                             histogram_freq=0,
                             write_graph=False,
                             write_images=False)

    val_x, val_y = generator.next(amount=val_amount)

    print("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=batch_size, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, tbCallBack], validation_data=(val_x, val_y))
