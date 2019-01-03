import numpy as np

import time
import gc
import keras
from keras import backend as K 
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, SpatialDropout1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Flatten
from keras.utils import np_utils
import pickle
import json

import getbatch_binary as getbatch


def data():
    return getbatch.generator, getbatch.val_generator

def model():
    K.clear_session()
    CLASS = 0
    num_classes = 1 #True/False
    IMSIZE = 128

    # 1D CNN neural network
    model_m = Sequential()
    model_m.add(Reshape((IMSIZE, IMSIZE,1),  input_shape=(IMSIZE,IMSIZE)))
    
    model_m.add(Conv2D({{choice([32,64,128,256,512,1024])}}, kernel_size = {{choice([3,5,7,9])}}, activation='relu', padding = 'same'))
    model_m.add(MaxPooling2D(2))    
    model_m.add(Conv2D({{choice([32,64,128,256,512,1024])}}, kernel_size = {{choice([3,5,7,9])}}, activation='relu', padding = 'same'))
    model_m.add(MaxPooling2D(2))    
    model_m.add(Conv2D({{choice([64,128,256,512,1024])}}, kernel_size = 3, activation='relu', padding = 'same'))
    model_m.add(MaxPooling2D(2))    
    model_m.add(Conv2D({{choice([64,128,256,512,1024])}}, kernel_size = 3, activation='relu', padding = 'same'))
    model_m.add(MaxPooling2D(2))   
    model_m.add(Conv2D({{choice([64,128,256,512,1024])}}, kernel_size = 3, activation='relu', padding = 'same'))
    model_m.add(MaxPooling2D(2))    
    model_m.add(Conv2D({{choice([64,128,256,512,1024])}}, kernel_size = 3, activation='relu', padding = 'same'))
    model_m.add(MaxPooling2D(2))   
    model_m.add(Conv2D({{choice([64,128,256,512,1024])}}, kernel_size = 3, activation='relu', padding = 'same'))
    model_m.add(Flatten())

    if conditional({{choice(["dense","direct"])}}) != "direct":
        model_m.add(Dense({{choice([64,128,256,512,1024,2048])}}))
        model_m.add(Dropout({{uniform(0, 1)}}))
    model_m.add(Dense(num_classes, activation='sigmoid'))
    print(model_m.summary())

    # %%

    print("\n--- Fit the model ---\n")


    print("\nClass: " + getbatch.labels[CLASS])	
    # The EarlyStopping callback monitors training accuracy:
    # if it fails to improve for ten consecutive epochs,
    # training stops early
    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, restore_best_weights = True, verbose = 1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.000001)
    ]

    model_m.compile(loss='binary_crossentropy',
                    optimizer={{choice(["Adam","SGD","rmsprop"])}}, metrics=['accuracy'])

    # Hyper-parameters
    TEST_SIZE = 10000
    STEPS_PER_EPOCH = 100
    STEPS_PER_VAL = 50
    EPOCHS = 100

    res = model_m.fit_generator(getbatch.generator(EPOCHS*STEPS_PER_EPOCH,CLASS), epochs=EPOCHS, verbose=1,
                                    callbacks=callbacks_list, steps_per_epoch = STEPS_PER_EPOCH, 
                                    validation_data = getbatch.val_generator(EPOCHS * STEPS_PER_VAL,CLASS), 
                                    validation_steps=STEPS_PER_VAL)

    print("\n--- Check against test data ---\n")

    (x_test, y_test) = getbatch.getBatch(CLASS,TEST_SIZE,False)

    # Set input_shape / reshape for Keras

    score, acc = model_m.evaluate(x_test, y_test[:,CLASS], verbose=1)

    print("Accuracy on test data: %0.4f" % acc)
    print("Loss on test data: %0.4f" % score)

    return {'loss': -acc, 'status': STATUS_OK}

def run_trials():
    f= open("bestSettings","a+",1)
    f.write("\n")
    f.write("\n")
    f.write("Network_binary\n")

    K.clear_session()
    trials_step = 1
    max_trials = 21 # first run, prevents getting stuck in random search

    try:
        print("Loading")
        trials = pickle.load(open("my_model.hyperopt","rb"))
        print("Found save Trials! loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except Exception as e:  # create a new trials object and start searching
        print(e)
        trials = Trials()

    trainGen, testGen = data()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=max_trials,
                                          trials=trials,
                                          verbose=False)

    best = []
    acc = 0;
    for i in trials.trials:
        if (-i.get('result').get('loss')) > acc:
            acc = -i.get('result').get('loss') 
            best = i.get('misc').get('vals')

    print(acc)
    print(best)

    f.write("Acc on test data: %0.4f \n" % -acc)
    f.write(str(best))

    with open("my_model.hyperopt", "wb") as f2:
        pickle.dump(trials,f2)


if __name__ == '__main__':
    t = time.time()
    run_trials()
    print(time.time() - t)

    