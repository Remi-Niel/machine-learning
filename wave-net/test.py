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

label_set = [x[1] for x in os.walk("data/")][0]
label_set = sorted(label_set)

THRESHOLD = 0.5


def getinput(file_name):
	(sample_rate, signal) = wavfile.read(file_name)

	Nsamp = math.floor(signal.shape[0]/44100)*2

	inputs = np.zeros((int(Nsamp),44100))

	mono = signal.sum(axis = 1) / 2

	mean = np.mean(mono) # is about zero
	stddev = np.std(mono)
	if stddev == 0:
		stddev = 1

	mono = (mono - mean) / stddev

	for i in range(Nsamp-1):
		sample = mono[i*22050:(i + 2)*22050]
		inputs[i,:] = sample
	
	return inputs

directory = 'testing/'

sample_files = glob.glob("testing/*.wav", recursive = True)

model_files = glob.glob("models/*.model", recursive = True)

model = []

for m in range(len(model_files)):
	del model
	K.clear_session()

	model = load_model(model_files[m])

	label = model_files[m].split("/")[1]
	label = label.replace('.model','')

	print(label)

	sumG = 0
	sumF = 0
	Gcount = 0
	Fcount = 0

	TP = 0
	FP = 0
	TN = 0
	FN = 0

	for idx in progressbar.progressbar(range(len(sample_files))): 
		wav_file = sample_files[idx]

		txt_file=wav_file.replace(".wav",".txt")

		f = open(txt_file,"r")

		labels = f.read().splitlines()


		for i in range(len(labels)):
			labels[i] = ''.join(labels[i].split())

		input = getinput(wav_file)

		predictions = model.predict(input)

		mean = np.mean(predictions)
		prediction = mean > THRESHOLD
		ground_truth = label in labels

		if ground_truth:
			sumG += ground_truth
			Gcount += 1
		else:
			sumF += ground_truth
			Fcount += 1
		#	print(mean)

		correct = (prediction == ground_truth)

		if correct:
			if prediction:
				TP+=1
			else:
				TN+=1
		else:
			if prediction:
				FP+=1
			else:
				FN+=1
	print(TP)
	print(FP)
	print(TN)
	print(FN)

	print(sumG/Gcount)
	print(sumF/Fcount)

	print(label_set[m])
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	print("Precision: " +str(precision))
	print("Recall: " +str(recall))






