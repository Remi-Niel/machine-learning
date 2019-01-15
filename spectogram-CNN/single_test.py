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

	

	IMSIZE = 128

	inputs = np.zeros((int(Nsamp), IMSIZE, IMSIZE))


	for i in range(Nsamp) :
		subsample = mono[(i*44100):(i+1)*44100]

		spec = mlab.specgram(subsample, Fs = 44100)[0]

		spec = scipy.misc.imresize(spec, [IMSIZE,IMSIZE])

		inputs[i,:,:] = spec
	
	return inputs


# def getinput(file_name, nSamp = 100):
# 	(sample_rate, signal) = wavfile.read(file_name)
# 	mono = signal.sum(axis = 1) / 2

# 	mean = np.mean(mono) # is about zero
# 	stddev = np.std(mono)
# 	if stddev == 0:
# 		stddev = 1

# 	mono = (mono - mean) / stddev
	
# 	IMSIZE = 128

# 	inputs = np.zeros((nSamp, IMSIZE, IMSIZE))

# 	for i in range(nSamp):
# 		tmp = random.randint(0, len(mono)-44100 - 1)
# 		sample = mono[tmp:tmp+44100]
# 		spec = mlab.specgram(sample, Fs = 44100)[0]

# 		spec = scipy.misc.imresize(spec, [IMSIZE,IMSIZE])

# 		inputs[i,:,:] = spec
	
# 	return inputs

sample_files = glob.glob("testing/*.wav", recursive = True)

correct = 0;

model = load_model("multi_model/model.model")

label_set = getbatch.labels;

for idx in progressbar.progressbar(range(len(sample_files))): 
	wav_file = sample_files[idx]

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

	# print(summed)

	best = np.argmax(summed)

	label = label_set[best]

	# print(best)
	# print(label)
	# print(labels)

	if label in labels:
		correct += 1

	if(idx%100 == 0 and idx > 0):
		print(correct/idx)


acc = correct / len(sample_files);
print("Accuracy: " + str(acc)) 





