import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model

from models import fcn, unet
from data_loader import Generator


if __name__ == "__main__":
    # TODO: Load arguments from command line
    num_classes = 10
    input_size = 160
    epochs = 1000
    batch_size = 1000
    val_amount = 50


    generator = Generator(patch_size=input_size, batch_size=batch_size)

    model = fcn.fcn32(input_size=input_size, num_classes=num_classes)

    if os.path.isfile('weights/fcn.hdf5'):
        load = input("Saved weights found. Load? (y/n)")
        if load == "y":
            print("Loading saved weights")
            model.load_weights('weights/fcn.hdf5')
    model.summary()
    plot_model(model, show_shapes=False, to_file="images/fcn_model.png")
    plot_model(model, show_shapes=True, to_file="images/fcn_model_shapes.png")
    if not os.path.exists('weights'):
        os.makedirs('weights')
    model_checkpoint = ModelCheckpoint('weights/fcn.hdf5', monitor='loss', save_best_only=True)

    # Setup tensorboard model
    tbCallBack = TensorBoard(log_dir='tensorboard_log/', histogram_freq=1,
                             write_graph=True, write_images=True)

    val_x, val_y = generator.next(amount=val_amount)

    print("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=batch_size, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, tbCallBack], validation_data=(val_x, val_y))
