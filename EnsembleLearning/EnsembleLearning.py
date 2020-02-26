# Jadie Adams

import numpy as np
from random import randint
import math
import sys
sys.path.append("../DecisionTree/")
import DecisionTree

def getIntLabels(labels):
	int_labels = []
	for label in labels:
		if label == 'yes':
			int_labels.append(1)
		else:
			int_labels.append(-1)
	int_labels = np.array(int_labels)
	return int_labels

def AdaBoost(data, labels, test_data, test_labels, values, T, printError=False):
	# Step 1: Initialize weights
	weights = np.array([1/len(data)]*len(data))

	int_labels = getIntLabels(labels)
	int_test_labels = getIntLabels(test_labels)

	# Step 2 for each iteration...
	h_ts = []
	votes = []
	for iteration in range(T):
		h_t = DecisionTree.getDecisionStump(data,labels,values,weights)
		h_ts.append(h_t)
		error, predictions = DecisionTree.getErrorAndPredictions(h_t, data, labels)
		summa = 0
		for index in range(len(predictions)):
			summa += weights[index]*predictions[index]*int_labels[index]
		e_t = .5 - (.5*summa)
		if printError:
			error, test_predictions = DecisionTree.getErrorAndPredictions(h_t, test_data, test_labels)
			summa = 0
			for index in range(len(test_predictions)):
				summa += (1/len(test_labels))*test_predictions[index]*int_test_labels[index]
			test_error = .5 - (.5*summa)
			print(iteration, e_t, test_error)
		# compute it's vote
		vote_t = 0.5 * np.log((1-e_t)/e_t)
		votes.append(vote_t)
		# update the weights
		new_weights_0 = weights*np.exp(-vote_t*(int_labels*predictions))
		new_weights = new_weights_0/np.sum(new_weights_0)
		weights = new_weights
	# Step 3, return the final hypothesis:
	return h_ts, votes

def testAdaBoost(trees, votes, data, labels):
	int_labels = getIntLabels(labels)
	errors = 0
	for data_index in range(len(data)):
		example = data[data_index]
		label = int_labels[data_index]
		summation = 0
		for index in range(len(trees)):
			tree = trees[index]
			vote = votes[index]
			tree_pred = DecisionTree.traceback(example, tree)
			if tree_pred == "yes":
				tree_pred = 1
			else:
				tree_pred = -1
			summation += tree_pred*vote
		prediction = np.sign(summation)
		if prediction != label:
			errors += 1
	return errors/len(labels)

def Bagging(data, labels, values, M, T):
	trees = []
	for t in range(T):
		# get data subset
		data_subset = []
		labels_subset = []
		for m in range(M):
			index = randint(0, len(data)-1)
			data_subset.append(data[index])
			labels_subset.append(labels[index])
		data_subset = np.array(data_subset)
		labels_subset = np.array(labels_subset)
		# get model
		tree = DecisionTree.ID3(data_subset, labels_subset, values, "information_gain", 0, data.shape[1])
		trees.append(tree)
	return trees

def testBagging(trees, data, labels):
	int_labels = getIntLabels(labels)
	errors = 0
	for data_index in range(len(data)):
		example = data[data_index]
		label = int_labels[data_index]
		summation = 0
		for tree in trees:
			tree_pred = DecisionTree.traceback(example, tree)
			if tree_pred == "yes":
				tree_pred = 1
			else:
				tree_pred = -1
			summation += tree_pred
		prediction = np.sign(summation)
		if prediction != label:
			errors += 1
	return errors/len(labels)


def RandomForest(data, labels, values, M, attr_num, T):
	trees = []
	for t in range(T):
		# get data subset
		data_subset = []
		labels_subset = []
		for m in range(M):
			index = randint(0, len(data)-1)
			data_subset.append(data[index])
			labels_subset.append(labels[index])
		data_subset_0 = np.array(data_subset)
		labels_subset = np.array(labels_subset)
		# get attr subset
		indices = []
		while len(indices) < attr_num:
			ind = randint(0, len(values)-1)
			if ind not in indices:
				indices.append(ind)
		data_subset = data_subset_0
		values_subset = values
		rm_indices = []
		for index in range(len(values)):
			if index not in indices:
				rm_indices.append(index)
		data_subset = np.delete(data_subset_0, rm_indices, 1)
		values_subset = np.delete(values, rm_indices, 0)
		# get model
		tree = DecisionTree.ID3(data_subset, labels_subset, values_subset, "information_gain", 0, data_subset.shape[1])
		trees.append(tree)
	return trees

