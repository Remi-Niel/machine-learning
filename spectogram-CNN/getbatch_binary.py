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

sample_files = [];
NUM_DATAFILES = 0;
length_cat = [];
for i in range(NUM_LABELS):
	sample_files.append(glob.glob(directory+labels[i]+'/*.wav',recursive=True))
	shuffle(sample_files[i])
	NUM_DATAFILES += len(sample_files[i])
	length_cat.append(len(sample_files[i]))

print(length_cat)
def num_class():
	labels = [x[1] for x in os.walk(directory)][0] #['piano','violin']

	NUM_LABELS = len(labels)
	
	return NUM_LABELS

def one_hot(label_array,num_classes):
	x = label_array.astype(int)
	return np.squeeze(np.eye(num_classes)[x.reshape(-1)])

def getBatch(t, size = 100, train = True):
	start = 0
	end = 0.8
	if not train:
		start = end
		end = 1

	#print(label_indexes)

	data = np.zeros((int(size), IMSIZE, IMSIZE))
	labels = []

	for i in range(size):
		file_name = ""
		if random.randint(0,1):
			file_name = sample_files[t][random.randint(int(start * length_cat[t]), int(end * length_cat[t] - 1))]
		else:
			possible = list(range(NUM_LABELS));
			possible.remove(t);
			chosen = random.choice(possible);
			file_name = sample_files[chosen][random.randint(int(start * length_cat[chosen]), int(end * length_cat[chosen] - 1))]
			
				
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

def generator(n,i):
	for idx in range(n):
		(x,y) = getBatch(i)
		yield (x,y[:,i])

def val_generator(n,i):
	for idx in range(n):
		(x,y) = getBatch(i,train = False)
		yield (x,y[:,i])
		

#print(getBatch(1000,False))

