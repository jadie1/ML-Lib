import os
import numpy as np 
import LogisticRegression
import sys
sys.path.append("..")
import DataUtils

d = 3
init = .001
def lrf(t):
	return init/(1+((init/d)*t))


if __name__ == "__main__":
	dataFolder = "../Data/bank-note/"

	# Get Data
	train_data, train_labels, test_data, test_labels = DataUtils.getData(dataFolder)
	train_labels = train_labels.astype('float64')
	test_labels = test_labels.astype('float64')

	# Train
	weights = LogisticRegression.MLstochasticGradientDescent(lrf, train_data, train_labels)
	train_error = LogisticRegression.getError(weights, train_data, train_labels)
	print(train_error)
	test_error = LogisticRegression.getError(weights, test_data, test_labels)
	print(test_error)
