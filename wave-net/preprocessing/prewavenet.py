import os
import glob
import random
from random import shuffle
from scipy.io import wavfile
import time
import progressbar
import numpy

#Make sure you do this in a different function
#That way python garbage collection makes sure the large numpy array
#is deleted after this function is done reducing memory usage
def SamplesFromFile(file_name):
    (sample_rate, signal) = wavfile.read(file_name)
    signal_length = len(signal)
    for j in range(10000):    #randomly select 10000 samples from file
        start = random.randint(0, signal_length-24000 - 1) #select random point
        sample = signal[start : (start + 24000)]
        name = "data/"+ labels[x]+"/"+str(i * 10000 + j)+".npy"
        with open(name,'wb') as f:
            numpy.save(f, sample)
        bar.update(i * 1000 + j)
    return 

def one_hot(label_array,num_classes):
    return numpy.squeeze(numpy.eye(num_classes)[label_array.reshape(-1)])


SAMPLE_LENGTH = 24000 #half second of sound (assuming 48000 sample-rate which is the default of youtube-dl)

directory = 'samples/'

labels = [x[1] for x in os.walk(directory)][0] #['piano','violin']

labels = sorted(labels)     #consistend label numbers 
NUM_LABELS = len(labels)

label_indexes = {labels[i]: i for i in range(0,NUM_LABELS)} #testing labels
print(label_indexes);
signal=[];

print("Creating samples");
for x in range(NUM_LABELS):
    try:
        os.makedirs("data/"+labels[x]);
        print("Directory data/",labels[x]," created")
    except FileExistsError:
        print("Directory data/",labels[x]," already exists")
    sample_files = glob.glob(directory+'/'+labels[x]+'/*.wav',recursive=True)
    NUM_DATAFILES = len(sample_files)
    print("Creating " + labels[x] +" samples")
    bar = progressbar.ProgressBar(max_value=100000);
    for i in range(10): #randomly select 10 files
        file_name = sample_files[random.randint(0,NUM_DATAFILES - 1)]
        SamplesFromFile(file_name);

data_files = glob.glob("data/**/*.npy", recursive = True); #get filepaths

shuffle(data_files);

num_data_files = len(data_files);
print(num_data_files);

data_labels = [];


print("Generating labels");
bar = progressbar.ProgressBar(max_value=num_data_files);
i=0
for file in data_files:
    # file will be /data/[category]/name.wav so category is
    label = file.split('/')[1];
    data_labels.append(label_indexes[label]);
    i += 1
    bar.update(i)

assert num_data_files == len(data_labels);

data_labels = numpy.array(data_labels);

data_labels = one_hot(data_labels, NUM_LABELS);

TRAIN_TEST_SPLIT = 0.15 #Part of data used to test performance

nr_test_data = int(num_data_files * TRAIN_TEST_SPLIT)

train_data_files = data_files[nr_test_data:]
test_data_files = data_files[:nr_test_data]

train_labels = data_labels[nr_test_data:]
test_labels = data_labels[:nr_test_data]

assert len(train_data_files) + len(test_data_files) == num_data_files
assert len(train_labels) + len(test_labels) == num_data_files

with open("data/data.npz",'wb') as data_file:
    numpy.savez(data_file,
        train_data_files=train_data_files,
        test_data_files=test_data_files,
        train_labels=train_labels,
        test_labels=test_labels)

data = numpy.load("data/data.npz");
assert train_data_files[0] == data['train_data_files'][0];
