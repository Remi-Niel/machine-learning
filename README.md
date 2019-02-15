# machine-learning

Training:

To train the networks run network_multi.py or network_binary.py from the terminal in either 1d-conv or spectogram-CNN. Training data should be stored in 1d-conv/data or spectogram-CNN/data depending on which you are running. The structure should be the same as the IRMAS training set. Make sure you set the variable num_classes in network_multi.py or for CLASS in range(11) in network_binary.py.

trained models will be stored in a models folder.

Testing:
To test the networks run multi_test.py (test performance when recognizing all instruments), single_test.py (test performance when returning 1 instrument) from the terminal in either 1d-conv or spectogram-CNN.


