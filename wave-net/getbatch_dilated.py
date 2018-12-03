import os
import glob
import random
from random import shuffle
from scipy.io import wavfile
import time
import progressbar
import numpy as np

directory = 'data/'

def num_class():
	labels = [x[1] for x in os.walk(directory)][0] #['piano','violin']

	NUM_LABELS = len(labels)
	
	return NUM_LABELS

def one_hot(label_array,num_classes):
    return np.squeeze(np.eye(num_classes)[label_array.reshape(-1)])

def getBatch(size = 30, train = True):
	start = 0
	end = 0.8
	if not train:
		start = 0.8
		end = 1

	labels = [x[1] for x in os.walk(directory)][0] #['piano','violin']

	labels = sorted(labels)     #consistend label numbers 
	NUM_LABELS = len(labels)

	label_indexes = {labels[i]: i for i in range(0,NUM_LABELS)} #testing labels
	#print(label_indexes)

	data = []
	labels = []



	sample_files = glob.glob(directory+'/*/*.wav',recursive=True)
	shuffle(sample_files)
	NUM_DATAFILES = len(sample_files)

	for i in range(size):
		file_name = sample_files[random.randint(start * NUM_DATAFILES, end * NUM_DATAFILES - 1)]
		(sample_rate, signal) = wavfile.read(file_name)
		del sample_rate

		tmp = random.randint(0, len(signal)-44100 - 1)
		signal = signal[tmp:(tmp + 44100):4]

		mono = signal.sum(axis=1) / 2

		mean = 0
		stddev = np.std(mono)

		mono = (mono - mean) / stddev

		data.append(mono.reshape(11025,1))

		labels.append(label_indexes[file_name.split('/')[1]])
	
	labels = np.array(labels)

	return (np.array(data),one_hot(labels,NUM_LABELS))

def generator(n):
	for idx in range(n):
		(x,y) = getBatch()
		yield (x,y)
		

#print(getBatch(1000,False))

