from keras.callbacks import ModelCheckpoint
import numpy as np

from algorithms.unet.model import unet
from data_loader import Generator

if __name__ == "__main__":
    num_classes = 10
    input_size = 160
    epochs = 1
    batch_size = 2

    generator = Generator(patch_size=input_size, batch_size=batch_size)

    model = unet(input_size=input_size, num_classes=num_classes)
    model.summary()
    model_checkpoint = ModelCheckpoint('weights/unet.hdf5', monitor='loss', save_best_only=True)

    val_x, val_y = generator.next()

    print("Starting training")

    for i in range(epochs):
        train_x, train_y = generator.next()
        print(f"Input data shape: {train_x.shape}")
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint], validation_data=(val_x, val_y))