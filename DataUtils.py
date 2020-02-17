import os
import numpy as np
import sys
sys.path.append("DecisionTree/")
import DecisionTree

# Input: folder with train.csv and test.csv
# Output: data_names, label_name, train_data, train_labels, test_data, test_labels
def getData(folder):
	train = folder + '/train.csv'
	test = folder + '/test.csv'
	train_data, train_labels = getDataFromCSV(train)
	test_data, test_labels = getDataFromCSV(test)
	return train_data, train_labels, test_data, test_labels

# Input: csv file
# Output: data_names, label_name, data, labels
def getDataFromCSV(csvFile):
	data = []
	with open(csvFile, 'r') as f:
		for line in f:
			data.append(line.strip().split(','))
	all_data = np.array(data)
	data = all_data[:,:-1]
	labels = all_data[:,-1]
	return data, labels

# returns binarized data
# above median -> high
# below median -> low
def binarize(data, indices):
	binarized = data
	for index in indices:
		med = np.median(np.array(binarized[:,index]).astype(np.float))
		for example_index in range(data.shape[0]):
			if float(data[example_index][index]) > med:
				binarized[example_index][index] = 'high'
			else:
				binarized[example_index][index] = 'low'
	return binarized

# replace values of "unknown" with most common value
def replaceMissing(data, labels):
	for index in range(data.shape[1]):
		replace = DecisionTree.getMostCommonLabel(data[:,index][data[:,index] != "unknown"])
		data[:,index] = np.where(data[:,index] == "unknown", replace, data[:,index])
	replace = DecisionTree.getMostCommonLabel(labels[labels != "unknown"])
	labels = np.where(labels == "unknown", replace, labels)
	return data, labels