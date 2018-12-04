import numpy as np

import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, SpatialDropout1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Flatten
from keras.utils import np_utils

import getbatch_binary as getbatch

f= open("log","a+")
f.write("\n")
f.write("\n")
f.write("Network_binary")
num_classes = 2 #True/False

TIME_PERIODS = 44100
num_sensors = 1
input_shape = (TIME_PERIODS*num_sensors)

# 1D CNN neural network
model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors),  input_shape=(input_shape,)))
model_m.add(Conv1D(64, 2, strides = 2, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(64, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(128, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(128, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(256, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(256, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(512, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(512, 2, strides = 2, activation='relu'))
model_m.add(Flatten())
model_m.add(Dense(1024))
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())

# %%

print("\n--- Fit the model ---\n")

for CLASS in 11:

    # The EarlyStopping callback monitors training accuracy:
    # if it fails to improve for ten consecutive epochs,
    # training stops early
    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
    ]

    model_m.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # Hyper-parameters
    TEST_SIZE = 1000
    STEPS_PER_EPOCH = 1
    STEPS_PER_VAL = 1
    EPOCHS = 1

    res = model_m.fit_generator(getbatch.generator(EPOCHS*STEPS_PER_EPOCH,CLASS), epochs=EPOCHS, verbose=1,callbacks=callbacks_list, steps_per_epoch = STEPS_PER_EPOCH, validation_data = getbatch.val_generator(EPOCHS * STEPS_PER_VAL,CLASS), validation_steps=STEPS_PER_VAL)

    print("\n--- Check against test data ---\n")

    (x_test, y_test) = getbatch.getBatch(CLASS,TEST_SIZE,False)

    # Set input_shape / reshape for Keras

    score = model_m.evaluate(x_test, getbatch.one_hot(y_test[:,CLASS],2), verbose=1)

    f.write(CLASS)
    print("\nAccuracy on test data: %0.4f" % score[1])
    print("\nLoss on test data: %0.4f" % score[0])

    f.write(CLASS)
    f.write("\nAccuracy on test data: %0.4f" % score[1])
    f.write("\nLoss on test data: %0.4f" % score[0])
