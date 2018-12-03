# Compatibility layer between Python 2 and Python 3
import numpy as np

import keras
from keras.optimizers import SGD
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, add, multiply
from keras.layers import Conv2D, MaxPooling2D, Convolution1D, AtrousConvolution1D, Flatten, Activation, Input
from keras.utils import np_utils

import getbatch_dilated as getbatch

num_classes = getbatch.num_class()

TIME_PERIODS = 11025
num_sensors = 1
input_shape = (TIME_PERIODS*num_sensors)


def wavenetBlock(n_fil, fil_size, dilation_rate):
	def f(input_):
		residual = input_
		tanh_out = AtrousConvolution1D(n_fil, fil_size,
						atrous_rate=dilation_rate,
						border_mode='same',
						activation='tanh')(input_)
		sigmoid_out = AtrousConvolution1D(n_fil, fil_size,
						atrous_rate=dilation_rate,
						border_mode='same',
						activation='sigmoid')(input_)
		multiplied = multiply([tanh_out, sigmoid_out])
		skip_out = Convolution1D(1, 1, activation='relu', border_mode='same')(multiplied)
		out = add([skip_out, residual])
		return out, skip_out
	return f


def get_model(input_size):
	input_ = Input(shape=(input_size, 1))
	A, B = wavenetBlock(64, 2, 2)(input_)
	skip_connections = [B]
	for i in range(20):
		A, B = wavenetBlock(64, 2, 2**((i+2)%9))(A)
		skip_connections.append(B)
	net = add(skip_connections)
	net = Activation('relu')(net)
	net = Convolution1D(1, 1, activation='relu')(net)
	net = Convolution1D(1, 1)(net)
	net = Flatten()(net)
	net = Dense(num_classes, activation='softmax')(net)
	model = Model(input=input_, output=net)
	model.compile(loss='categorical_crossentropy', optimizer='Adam',
	  metrics=['accuracy'])
	model.summary()
	return model

# 1D CNN neural network
model_m = get_model(TIME_PERIODS)

# %%

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for ten consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='acc', patience=10)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='Adam', metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 10000
STEPS_PER_EPOCH = 100
EPOCHS = 3000

res = model_m.fit_generator(getbatch.generator(EPOCHS*STEPS_PER_EPOCH), epochs=EPOCHS, verbose=1,callbacks=callbacks_list, steps_per_epoch = STEPS_PER_EPOCH)

print("\n--- Check against test data ---\n")

(x_test, y_test) = getbatch.getBatch(BATCH_SIZE,False)

# Set input_shape / reshape for Keras

score = model_m.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy on test data: %0.4f" % score[1])
print("\nLoss on test data: %0.4f" % score[0])

# %%

print("\n--- Confusion matrix for test data ---\n")

# %%

print("\n--- Classification report for test data ---\n")
