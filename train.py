import os

from keras.callbacks import ModelCheckpoint, TensorBoard

from models.unet import unet
from data_loader import Generator


if __name__ == "__main__":
    # TODO: Load arguments from command line
    num_classes = 10
    input_size = 160
    epochs = 1
    batch_size = 1
    val_amount = 1

    generator = Generator(patch_size=input_size, batch_size=batch_size)

    model = unet(input_size=input_size, num_classes=num_classes)

    if os.path.isfile('weights/unet.hdf5'):
        load = input("Saved weights found. Load? (y/n)")
        if load == "y":
            print("Loading saved weights")
            model.load_weights('weights/unet.hdf5')
    model.summary()
    if not os.path.exists('weights'):
        os.makedirs('weights')
    model_checkpoint = ModelCheckpoint('weights/unet.hdf5', monitor='loss', save_best_only=True)

    # Setup tensorboard model
    tbCallBack = TensorBoard(log_dir='tensorboard_log/', histogram_freq=1,
                             write_graph=True, write_images=True)

    val_x, val_y = generator.next(amount=val_amount)

    print("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=batch_size, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, tbCallBack], validation_data=(val_x, val_y))
