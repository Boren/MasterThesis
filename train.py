import os

from keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import numpy as np

from models.unet import unet
from data_loader import Generator, CLASS_TO_LABEL

if __name__ == "__main__":
    # TODO: Load arguments from command line
    num_classes = 10
    input_size = 160
    epochs = 1000
    batch_size = 5

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

    val_x, val_y = generator.next(amount=5)

    print("Starting training")

    model.fit_generator(generator.generator(), steps_per_epoch=batch_size, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, tbCallBack], validation_data=(val_x, val_y))

    print("Doing some sample predicitons")

    test_amount = 3
    test_x, test_y = generator.next(amount=3)

    test_y_result = model.predict(test_x, batch_size=1)

    # Save results
    for cls in range(10):
        for patchnum in range(3):
            if np.amax(test_y[patchnum, :, :, cls]) == 1:
                plt.figure()
                ax1 = plt.subplot(131)
                ax1.set_title('Raw RGB data')
                ax1.imshow(test_x[patchnum, :, :, :], cmap=plt.get_cmap('gist_ncar'))

                ax2 = plt.subplot(132)
                ax2.set_title('Ground Truth ({cls})'.format(cls=CLASS_TO_LABEL[cls+1]))
                ax2.imshow(test_y[patchnum, :, :, cls], cmap=plt.get_cmap('gray'))

                ax3 = plt.subplot(133)
                ax3.set_title('Prediction ({cls})'.format(cls=CLASS_TO_LABEL[cls+1]))
                ax3.imshow(test_y_result[patchnum, :, :, cls], cmap=plt.get_cmap('gray'),
                           interpolation='nearest', vmin=0, vmax=1)
                plt.show()
