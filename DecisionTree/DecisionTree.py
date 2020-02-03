# Jadie Adams
import numpy as np
import math

def getMostCommonLabel(labels):
	unique,pos = np.unique(labels,return_inverse=True) 
	counts = np.bincount(pos)
	maxpos = counts.argmax()
	return(unique[maxpos])

def getBestAttribute(data, labels, heuristic):
	# define heuristic call
	if heuristic == "information_gain":
		heuristic_call = getEntropy
	elif heuristic == "gini_index":
		heuristic_call = getGiniIndex
	elif heuristic == "majority_error":
		heuristic_call = getMajorityError
	else:
		print("Error heuristic unimplemented. \n Please use information_gain, gini_index, or majority_error.")
	# get total value
	total_entropy = heuristic_call(labels)
	max_gain = 0
	# loop through attributes and get gain for each
	attr_index = 0
	for attr_data in data.T:
		expected_entropy = 0
		values,pos = np.unique(attr_data,return_inverse=True)
		# loop though values to calculate expected entropy
		for value_index in range(values.shape[0]):
			indices = np.where(pos == value_index)
			value_labels = labels[indices]
			value_frac = value_labels.shape[0]/labels.shape[0]
			value_entropy = heuristic_call(value_labels)
			expected_entropy += value_frac*value_entropy
		attr_gain = total_entropy - expected_entropy
		if attr_gain > max_gain:
			max_gain = attr_gain
			best_attr = attr_index
		attr_index += 1
	return attr_index

#### Heurisitcs
def getEntropy(labels):
	entropy = 0
	unique,pos = np.unique(labels,return_inverse=True)
	counts = np.bincount(pos)
	probs = counts/np.sum(counts)
	for prob in probs:
		entropy += -(prob)*math.log(prob,2)
	return entropy

def getGiniIndex(labels):
	GI = 0
	unique,pos = np.unique(labels,return_inverse=True)
	counts = np.bincount(pos)
	probs = counts/np.sum(counts)
	for prob in probs:
		GI += prob**2
	return 1-GI

def getMajorityError(labels):
	unique,pos = np.unique(labels,return_inverse=True)
	counts = np.bincount(pos)
	maxCount = np.max(counts)
	sumCounts = np.sum(counts)
	ME = (sumCounts - maxCount)/sumCounts
	return ME



def ID3(data, labels, heurisitic, max_depth):
	node = {}
	# base case
	if np.unique(labels).shape[0] == 1:
		print("base case")
		node['label'] = labels[0]
		return node
	# out of attributes
	elif data.shape == 0:
		print("out of attributes")
		node['label'] = getMostCommonLabel(labels)
		return node
	else:
		print("get best attr")
		attr = getBestAttribute(data, labels, heurisitic)
		print(attr)
		return node
		


