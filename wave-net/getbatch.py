import os
import glob
import random
from random import shuffle
from scipy.io import wavfile
import time
import progressbar
import numpy as np

def one_hot(label_array,num_classes):
    return np.squeeze(np.eye(num_classes)[label_array.reshape(-1)])

def getBatch(size = 100):
	directory = 'samples/'

	labels = [x[1] for x in os.walk(directory)][0] #['piano','violin']

	labels = sorted(labels)     #consistend label numbers 
	NUM_LABELS = len(labels)

	label_indexes = {labels[i]: i for i in range(0,NUM_LABELS)} #testing labels
	print(label_indexes)

	data = np.zeros(100,44100 * 3)
	labels = []



	sample_files = glob.glob(directory+'/*/*.wav',recursive=True)
    NUM_DATAFILES = len(sample_files)

	for i in range(100):
		file_name = sample_files[random.randint(0,NUM_DATAFILES - 1)]
		data[i,:] = wavfile.read(file_name);
		labels.append(label_indexes[file_name.split('/')[1]]);


	return (data,one_hot(labels,NUM_LABELS))

print(getBatch(2)[1]);

