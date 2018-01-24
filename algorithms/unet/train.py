import os

from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

from algorithms.unet.model import unet
from data_loader import Generator, CLASS_TO_LABEL

if __name__ == "__main__":
    num_classes = 10
    input_size = 160
    epochs = 100
    batch_size = 3

    generator = Generator(patch_size=input_size, batch_size=batch_size)

    model = unet(input_size=input_size, num_classes=num_classes)
    model.summary()
    model_checkpoint = ModelCheckpoint('weights/unet.hdf5', monitor='loss', save_best_only=True)

    val_x, val_y = generator.next()

    print("Starting training")

    for i in range(epochs):
        train_x, train_y = generator.next()
        print(f"Input data shape: {train_x.shape}")
        print(f"Input data shape: {train_y.shape}")
        model.fit(train_x, train_y, batch_size=1, epochs=1, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint], validation_data=(val_x, val_y))

    print("Doing some sample predicitons")

    test_amount = 10
    test_x, test_y = generator.next(amount=4)

    test_y_result = model.predict(test_x, batch_size=1)

    # Save results
    for cls in range(10):
        for patchnum in range(10):
            if np.amax(test_y[patchnum, :, :, cls]) == 1:
                plt.figure()
                ax1 = plt.subplot(131)
                ax1.set_title('Raw RGB data')
                ax2 = plt.subplot(132)
                ax3 = plt.subplot(133)
                ax1.imshow(test_x[patchnum, :, :, :], cmap=plt.get_cmap('gist_ncar'))
                ax2.set_title(f'Ground Truth ({CLASS_TO_LABEL[cls+1]})')
                ax2.imshow(test_y[patchnum, :, :, cls], cmap=plt.get_cmap('gray'))
                ax3.set_title(f'Prediction ({CLASS_TO_LABEL[cls+1]})')
                ax3.imshow(test_y_result[patchnum, :, :, cls], cmap=plt.get_cmap('gray'))
                plt.show()
                input("Press Enter to continue...")
