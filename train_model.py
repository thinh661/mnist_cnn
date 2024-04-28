import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Convolution2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from matplotlib import pyplot as plt

print(os.listdir('/'))
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train / 255
X_test = X_test / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
model = Sequential()

model.add(
    Conv2D(filters=24, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=5e-4))
reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                              patience=3,
                              verbose=1,
                              factor=0.2,
                              min_lr=1e-6)

os.makedirs('./log', exist_ok=True)
log = keras.callbacks.TensorBoard(log_dir='./log',
                                  histogram_freq=1,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') is None:
            print("1")
        else:
            if (logs.get('acc') > 0.995):
                print("\nReached 99.5% accuracy so cancelling training!")
                self.model.stop_training = True


callbacks = myCallback()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2)
datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=100), steps_per_epoch=len(X_train) / 100,
                              epochs=5, validation_data=(X_test, y_test), callbacks=[callbacks])

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training Loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)

score = model.evaluate(X_test, y_test, batch_size=32)
print('score: ', score)
model.save('model_v2.h5')
