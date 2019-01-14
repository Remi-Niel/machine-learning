import os
import glob
import sys
import progressbar
import numpy as np
from scipy.io import wavfile
import math
import keras
import random
from keras.models import load_model
from keras import backend as K 
from tensorflow import Session

label_set = [x[1] for x in os.walk("data/")][0]
label_set = sorted(label_set)

THRESHOLD = 0.5

def determineOptimalThreshold(groundTmean, groundFmean):
	f = 0
	best_TP = -1;
	best_FP = -1;
	best_TN = -1;
	best_FN = -1;
	best_thresh = 0
	for t in np.linspace(0.1,.9,81):
		TP = sum(1 for x in groundTmean if x >= t)
		FN = len(groundTmean) - TP
		FP = sum(1 for x in groundFmean if x >= t)
		TN = len(groundFmean) - FP

		precision = 0;
		recall = 0;

		if(FP == 0):
			precision = 1.0
		else:
			precision = TP / (TP + FP)
			

		if(FN == 0):
			recall = 1.0
		else:
			recall = TP / (TP + FN)
			
		score = precision * recall;
		if (score > f):
			best_TP = TP;
			best_FP = FP;
			best_TN = TN;
			best_FN = FN;
			best_thresh = t
			f = score 

	precision = best_TP / (best_TP + best_FP)
	recall = best_TP / (best_TP + best_FN)
	print(best_thresh)
	print("Precision: " +str(precision))
	print("Recall: " +str(recall))

	return best_thresh

def getinput(file_name, Nsamp = 5):
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

directory = 'testing/'

sample_files = glob.glob("testing/*.wav", recursive = True)

model_files = glob.glob("models/*.model", recursive = True)

correct = 0;

model = load_model("multi_model/model.model")

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

	print(prediction)


	# label = model_files[bestGuess].split("/")[1]
	# label = label.replace('.model','')
	# if label in labels:
	# 	correct += 1

acc = correct / len(sample_files);
print("Accuracy: " + str(acc)) 





