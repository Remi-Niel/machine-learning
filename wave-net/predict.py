import os
import glob
import sys
import progressbar
import numpy as np
from scipy.io import wavfile
import scipy
from matplotlib import mlab
import math
import keras
import random
from keras.models import load_model
from keras import backend as K 
from tensorflow import Session
import getbatch_binary as getbatch


def getinput(file_name, Nsamp = 100):
	(sample_rate, signal) = wavfile.read(file_name)

	Nsamp = min(math.floor(signal.shape[0]/44100), Nsamp) 

	mono = signal.sum(axis = 1) / 2

	mean = np.mean(mono) # is about zero
	stddev = np.std(mono)
	if stddev == 0:
		stddev = 1

	mono = (mono - mean) / stddev

	inputs = np.resize(mono,Nsamp*44100).reshape(-1,44100)
	
	return inputs


model = load_model("multi_model/model.model")

label_set = getbatch.labels;

wav_file = sys.argv[1];
txt_file=wav_file.replace(".wav",".txt")
f = open(txt_file,"r")

labels = f.read().splitlines()


for i in range(len(labels)):
	labels[i] = ''.join(labels[i].split())

input = getinput(wav_file)

highestActivation = -1;
bestGuess = -1;

prediction = model.predict(input)

summed = np.sum(prediction, axis = 0)


best = np.argmax(summed)

label = label_set[best]

print("Recognised instrument: " + label)
print("Certainty: " + str(best/np.sum(summed)))

print("Contains: " + str(labels))
