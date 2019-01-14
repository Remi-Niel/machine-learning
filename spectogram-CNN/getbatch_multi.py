import os
import glob
import random
from random import shuffle
from scipy.io import wavfile
import scipy
from matplotlib import mlab
import time
import progressbar
import numpy as np
import sys


directory = 'data/'
IMSIZE = 128

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

def getBatch(size = 150, train = True):
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

	data = np.zeros((size,44100))
	labels = []



	sample_files = glob.glob(directory+'/*/*.wav',recursive=True)
	shuffle(sample_files)
	NUM_DATAFILES = len(sample_files)

	for i in range(size):
		file_name = sample_files[random.randint(start * NUM_DATAFILES, end * NUM_DATAFILES - 1)]
		
		(sample_rate, signal) = wavfile.read(file_name)
		del sample_rate

		mono = signal.sum(axis=1) / 2

		tmp = random.randint(0, len(signal)-44100 - 1)
		mono = mono[tmp:tmp+44100]

		spec = mlab.specgram(mono, Fs = 44100)[0]

		spec = scipy.misc.imresize(spec, [IMSIZE,IMSIZE])

		data[i,:,:] = spec

		labels.append(label_indexes[file_name.split('/')[1]])
	
	labels = np.array(labels)

	return (data,one_hot(labels,NUM_LABELS))
	
def generator(n):
	for idx in range(n):
		(x,y) = getBatch()
		yield (x,y)

def val_generator(n):
	for idx in range(n):
		(x,y) = getBatch(train = False)
		yield (x,y)
		

#print(getBatch(1000,False))

