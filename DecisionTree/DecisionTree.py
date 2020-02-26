# Jadie Adams
import numpy as np
import math
from random import randint

### ID3 Helpers ##########################################################################
def getMostCommonLabel(labels):
	unique,pos = np.unique(labels,return_inverse=True) 
	counts = np.bincount(pos)
	maxpos = counts.argmax()
	return(unique[maxpos])

def getDataSubset(data, labels, values, attr_index, value):
	indices = np.where(data[:,attr_index] == value)
	subset_data = data[indices,:][0]
	subset_data = np.delete(subset_data, attr_index, 1)
	subset_labels = labels[indices]
	subset_values = np.delete(values, attr_index, 0)
	return subset_data, subset_labels, subset_values

def getBestAttribute(data, labels, heuristic, weights=[]):
	# define heuristic call
	if heuristic == "information_gain":
		heuristic_call = getEntropy
	elif heuristic == "gini_index":
		heuristic_call = getGiniIndex
	elif heuristic == "majority_error":
		heuristic_call = getMajorityError
	elif heuristic == "weightedEntropy":
		heuristic_call = getWeightedEntropy
	else:
		print("Error heuristic unimplemented. \n Please use information_gain, gini_index, or majority_error.")
	# get total value
	if heuristic == "weightedEntropy":
		total_entropy = heuristic_call(labels, weights)
	else:
		total_entropy = heuristic_call(labels)
	max_gain = -1
	# loop through attributes and get gain for each
	attr_index = 0
	for attr_data in data.T:
		expected_entropy = 0
		values,pos = np.unique(attr_data,return_inverse=True)
		# loop though values to calculate expected entropy
		for value_index in range(values.shape[0]):
			indices = np.where(pos == value_index)
			value_labels = labels[indices]
			if heuristic == "weightedEntropy":
				value_weights = weights[indices]
				value_entropy = heuristic_call(value_labels, value_weights)
				value_frac = np.sum(value_weights)/1
			else:
				value_entropy = heuristic_call(value_labels)
				value_frac = value_labels.shape[0]/labels.shape[0]
			expected_entropy += value_frac*value_entropy
		attr_gain = total_entropy - expected_entropy
		if attr_gain >= max_gain:
			max_gain = attr_gain
			best_attr = attr_index
		attr_index += 1
	# print(max_gain, best_attr)
	return best_attr

#### Heurisitcs
def getEntropy(labels):
	entropy = 0
	unique,pos = np.unique(labels,return_inverse=True)
	counts = np.bincount(pos)
	probs = counts/np.sum(counts)
	for prob in probs:
		if prob != 0:
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

################### ID3 Algorithm ####################################################################
def ID3(data, labels, values, heurisitic, current_depth, max_depth):
	node = {}
	# base case, all remaining labels are the same
	if np.unique(labels).shape[0] == 1:
		node['label'] = labels[0]
		return node
	# out of attributes or max depth reached add most common label
	elif data.shape[0] == 0 or current_depth == max_depth:
		node['label'] = getMostCommonLabel(labels)
		return node
	else:
		current_depth += 1
		attr_index = getBestAttribute(data, labels, heurisitic)
		name = values[attr_index][0]
		node[name] = {}
		possible_values = values[attr_index][1]
		unique_values = np.unique(data[:, attr_index])
		for value in possible_values:
			# if no examples add most common label
			if value not in unique_values:
				node[name][value] = {'label':getMostCommonLabel(labels)}
			# else recursive call
			else:
				subset_data, subset_labels, subset_values = getDataSubset(data, labels, values, attr_index, value)
				node[name][value] = ID3(subset_data, subset_labels, subset_values, heurisitic, current_depth, max_depth)
		return node

############### Testing functions ######################################################################################

# test tree helper
def traceback(example, tree):
	if list(tree.keys())[0] == "label":
		return tree["label"]
	else:
		key = list(tree.keys())[0]
		value = example[key]
		subtree = tree[key][value]
		label = traceback(example, subtree)
		return label

def testTree(tree, data, labels):
	correct = 0
	for index in range(len(data)):
		example = data[index]
		label = labels[index]
		prediction = traceback(example, tree)
		if prediction == label:
			correct += 1
	accuracy = correct / len(labels)
	return accuracy
	
def getErrorAndPredictions(tree, data, labels):
	incorrect = 0
	predictions = []
	for index in range(len(data)):
		example = data[index]
		label = labels[index]
		prediction = traceback(example, tree)
		if prediction == "yes":
			predictions.append(1)
		else:
			predictions.append(-1)
		if prediction != label:
			incorrect += 1
	error = incorrect / len(labels)
	return error, np.array(predictions)

################# Weighted Decision Stump #########################################################################################
def getWeightedEntropy(labels,weights):
	entropy = 0
	unique,pos = np.unique(labels,return_inverse=True)
	probs = [0,0]
	for index in range(len(weights)):
		if pos[index] == 0:
			probs[0] += weights[index]
		else:
			probs[1] += weights[index]
	for prob in probs:
		if prob != 0:
			entropy += -(prob)*math.log(prob,2)
	return entropy

def getDecisionStump(data, labels, values, weights):
	node = {}
	heurisitic = "weightedEntropy"
	attr_index = getBestAttribute(data, labels, heurisitic, weights)
	name = values[attr_index][0]
	node[name] = {}
	for value in values[attr_index][1]:
		chosen_label = weight_label(data, labels, values, weights, attr_index, value)
		node[name][value] = {'label':chosen_label}
	return node

def weight_label(data, labels, values, weights, attr_index, value):
	indices = np.where(data[:,attr_index] == value)
	subset_labels = labels[indices]
	subset_weights = weights[indices]
	unique,pos = np.unique(subset_labels,return_inverse=True)
	weight0 = 0
	weight1 = 0
	for index in range(len(subset_labels)):
		sub_label = subset_labels[index]
		if sub_label == unique[0]:
			weight0 += subset_weights[index]
		elif sub_label == unique[1]:
			weight1 += subset_weights[index]
	if weight0 > weight1:
		chosen_label = unique[0]
	else:
		chosen_label = unique[1]
	return chosen_label