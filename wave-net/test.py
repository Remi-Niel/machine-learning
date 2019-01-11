import os
import glob
import sys
import progressbar
import numpy as np
from scipy.io import wavfile
import math
import keras
from keras.models import load_model
from keras import backend as K 


def getinput(file_name):
	(sample_rate, signal) = wavfile.read(file_name)

	print(signal.shape)
	Nsamp = math.floor(signal.shape[0]/44100)

	inputs = np.zeros((int(Nsamp),44100))

	mono = signal.sum(axis = 1) / 2

	mean = np.mean(mono) # is about zero
	stddev = np.std(mono)
	if stddev == 0:
		stddev = 1

	mono = (mono - mean) / stddev

	print(len(mono))

	for i in range(Nsamp):
		sample = signal[Nsamp*44100:(Nsamp + 1)*44100 - 1]
		print(len(signal))
		inputs[i,:] = sample
	
	return inputs

directory = 'testing/'

sample_files = glob.glob("testing/*.wav", recursive = True)

model_files = glob.glob("models/*.model", recursive = True)

model = []

for model_file in model_files:
	del model
	K.clear_session()

	model = load_model(model_file)

	for idx in progressbar.progressbar(range(len(sample_files))): 
		wav_file = sample_files[idx]

		txt_file=wav_file.replace(".wav",".txt")

		f = open(txt_file,"r")

		labels = f.read().splitlines()


		for i in range(len(labels)):
			labels[i] = ''.join(labels[i].split())

		input = getinput(wav_file)













