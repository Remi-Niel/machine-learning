import os
import glob
import random
from random import shuffle
from scipy.io import wavfile
import time
import progressbar
import numpy as np

directory = 'data/'

labels = [x[1] for x in os.walk(directory)][0] #['piano','violin']
labels = sorted(labels)     #consistend label numbers 
NUM_LABELS = len(labels)
label_indexes = {labels[i]: i for i in range(0,NUM_LABELS)} #testing labels

sample_files = glob.glob(directory+'/*/*.wav',recursive=True)
shuffle(sample_files)
NUM_DATAFILES = len(sample_files)

def num_class():
	labels = [x[1] for x in os.walk(directory)][0] #['piano','violin']

	NUM_LABELS = len(labels)
	
	return NUM_LABELS

def one_hot(label_array,num_classes):
	x = label_array.astype(int)
	return np.squeeze(np.eye(num_classes)[x.reshape(-1)])

def getBatch(t, size = 150, train = True):
	start = 0
	end = 0.8
	if not train:
		start = 0.8
		end = 1

	#print(label_indexes)

	data = np.zeros((int(size),44100))
	labels = []

	for i in range(size):
		file_name = sample_files[random.randint(start * NUM_DATAFILES, end * NUM_DATAFILES - 1)]
		if random.randint(0,1):
			while label_indexes[file_name.split('/')[1]] != t:
				file_name = sample_files[random.randint(start * NUM_DATAFILES, end * NUM_DATAFILES - 1)]
		else:
			while label_indexes[file_name.split('/')[1]] == t:
				file_name = sample_files[random.randint(start * NUM_DATAFILES, end * NUM_DATAFILES - 1)]
			
				
		(sample_rate, signal) = wavfile.read(file_name)
		del sample_rate

		mono = signal.sum(axis=1) / 2

		tmp = random.randint(0, len(signal)-44100 - 1)
		mono = mono[tmp:tmp+44100]

		mean = np.mean(mono)
		stddev = np.std(mono)

		mono = (mono - mean) / stddev

		data[i,:] = mono

		labels.append(label_indexes[file_name.split('/')[1]])
	
	labels = np.array(labels)

	return (data,one_hot(labels,NUM_LABELS))

def generator(n,i):
	for idx in range(n):
		(x,y) = getBatch(i)
		yield (x,one_hot(y[:,i],2))

def val_generator(n,i):
	for idx in range(n):
		(x,y) = getBatch(100,False)
		yield (x,one_hot(y[:,i],2))
		

#print(getBatch(1000,False))

