# Compatibility layer between Python 2 and Python 3
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Flatten
from keras.utils import np_utils

import getbatch

num_classes = getbatch.num_class()

TIME_PERIODS = 3* 44100 -1
num_sensors = 1
input_shape = (TIME_PERIODS*num_sensors)

# 1D CNN neural network
model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors),  input_shape=(input_shape,)))
model_m.add(Conv1D(32, 2, strides = 2, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(32, 2, strides = 2, activation='relu'))
model_m.add(Conv1D(64, 2, strides = 2, activation='relu'))
model_m.add(Conv1D(64, 2, strides = 2, activation='relu'))
model_m.add(Conv1D(128, 2, strides = 2, activation='relu'))
model_m.add(Flatten())
model_m.add(Dense(500))
model_m.add(Dropout(0.5))
model_m.add(Dense(1, activation='softmax'))
print(model_m.summary())

# %%

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True)
]

model_m.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 1000
EPOCHS = 100

res = model_m.fit_generator(getbatch.getBatch, epochs=EPOCHS, verbose=1, callbacks=callbacks_list, steps_per_epoch = 1)

print("\n--- Check against test data ---\n")

(x_test, y_test) = getbatch.getBatch(BATCH_SIZE,False)

# Set input_shape / reshape for Keras
x_test = x_test.astype("float32")

y_test = y_test.astype("float32")
y_test = y_test[:,2]

score = model_m.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

# %%

print("\n--- Confusion matrix for test data ---\n")

# %%

print("\n--- Classification report for test data ---\n")
