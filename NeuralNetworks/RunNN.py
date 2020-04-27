# Jadie Adams
import os
import numpy as np 
import NN
import sys
sys.path.append("..")
import DataUtils
 
d = 100
init = 1
def lrf(t):
	return init/(1+((init/d)*t))

if __name__ == "__main__":
	dataFolder = "../Data/bank-note/"

	# Get Data
	train_data, train_labels, test_data, test_labels = DataUtils.getData(dataFolder)
	train_labels = train_labels.astype('float64')
	test_labels = test_labels.astype('float64')

	epochs = 7
	print(str(epochs) + " epochs.")

	hidden_dims = [5,10,25,50,100]
	for hid_dim in hidden_dims:
		print("Hidden dim width = " + str(hid_dim))
		parameters = NN.train(hid_dim, epochs, lrf, train_data, train_labels)
		train_accuracy = NN.test(hid_dim, train_data, train_labels, parameters)
		print("    Train_accuracy: " + str(train_accuracy))
		test_accuracy = NN.test(hid_dim, test_data, test_labels, parameters)
		print("    Test accuracy: " + str(test_accuracy))


