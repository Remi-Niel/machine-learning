import numpy as np

import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, SpatialDropout1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Flatten
from keras.utils import np_utils

import getbatch_binary as getbatch

f= open("log","a+",1)
f.write("\n")
f.write("\n")
f.write("Network_binary")
num_classes = 1 #True/False
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

model_m.add(Conv1D(32, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))

model_m.add(Conv1D(32, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))

model_m.add(Conv1D(64, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))

model_m.add(Conv1D(32, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))

model_m.add(Conv1D(32, 2, strides = 2, activation='relu'))
model_m.add(MaxPooling1D(2))

model_m.add(Conv1D(256, 2, strides = 2, activation='relu'))
model_m.add(Flatten())

model_m.add(Dense(512))
model_m.add(Dropout(0.001))
model_m.add(Dense(num_classes, activation='sigmoid'))

print(model_m.summary())

# %%

print("\n--- Fit the model ---\n")

for CLASS in range(11):
    print("\nClass: " + getbatch.labels[CLASS])	
    # The EarlyStopping callback monitors training accuracy:
    # if it fails to improve for ten consecutive epochs,
    # training stops early
    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_acc', patience=20, restore_best_weights = True, verbose = 1),
	keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
    ]

    model_m.compile(loss='binary_crossentropy',
                    optimizer="Adam", metrics=['accuracy'])

    # Hyper-parameters
    TEST_SIZE = 10000
    STEPS_PER_EPOCH = 100
    STEPS_PER_VAL = 100
    EPOCHS = 1000

    res = model_m.fit_generator(getbatch.generator(EPOCHS*STEPS_PER_EPOCH,CLASS), epochs=EPOCHS, verbose=1,callbacks=callbacks_list, steps_per_epoch = STEPS_PER_EPOCH, validation_data = getbatch.val_generator(EPOCHS * STEPS_PER_VAL,CLASS), validation_steps=STEPS_PER_VAL)

    print("\n--- Check against test data ---\n")

    (x_test, y_test) = getbatch.getBatch(CLASS,TEST_SIZE,False)

    # Set input_shape / reshape for Keras

    score = model_m.evaluate(x_test, y_test[:,CLASS], verbose=1)

    print("Accuracy on test data: %0.4f" % score[1])
    print("Loss on test data: %0.4f" % score[0])

    f.write("\n\nClass: " + getbatch.labels[CLASS])
    f.write("\nAccuracy on test data: %0.4f" % score[1])
    f.write("\nLoss on test data: %0.4f" % score[0])

    model_m.save("models/" + getbatch.labels[CLASS]+".model")

