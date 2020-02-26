# Jadie Adams
# Example of ensemble learning using bank data
import numpy as np
import sys
from random import randint
import EnsembleLearning
sys.path.append("../DecisionTree/")
import DecisionTree
sys.path.append("..")
import DataUtils

if __name__ == "__main__":
	# Get data
	train_data, train_labels, test_data, test_labels = DataUtils.getData("../Data/bank")
	train_values = [
		[0,np.array(['low', 'high'])],
		[1,np.array(["admin.","unknown","unemployed","management","housemaid","entrepreneur",
			"student","blue-collar","self-employed","retired","technician","services"])],
		[2,np.array(["married","divorced","single"])],
		[3,np.array(["unknown","secondary","primary","tertiary"])],
		[4,np.array(["yes","no"])],
		[5,np.array(['low', 'high'])],
		[6,np.array(["yes","no"])],
		[7,np.array(["yes","no"])],
		[8,np.array(["unknown","telephone","cellular"])],
		[9,np.array(['low', 'high'])],
		[10,np.array(["jan", "feb", "mar","apr","may","jun","jul","aug", "sep", "oct","nov", "dec"])],
		[11,np.array(['low', 'high'])],
		[12,np.array(['low', 'high'])],
		[13,np.array(['low', 'high'])],
		[14,np.array(['low', 'high'])],
		[15,np.array(["unknown","other","failure","success"])]
		]
	train_data = DataUtils.binarize(train_data, [0,5,9,11,12,13,14])
	test_data = DataUtils.binarize(test_data, [0,5,9,11,12,13,14])
	int_labels = EnsembleLearning.getIntLabels(test_labels)

################################ Part A #######################################################
	print("Part A")
	iterations = [1,5,10,50,100,500,1000]
	print("# iterations, train_error, test_error")
	for iteration in iterations:
		hyps, votes = EnsembleLearning.AdaBoost(train_data, train_labels, test_data, test_labels, train_values, iteration)
		train_error = EnsembleLearning.testAdaBoost(hyps, votes, train_data, train_labels)
		test_error = EnsembleLearning.testAdaBoost(hyps, votes, test_data, test_labels)
		print(iteration, train_error, test_error)

	print("\nTraining and test error at each step:")
	hyps,votes = EnsembleLearning.AdaBoost(train_data, train_labels, test_data, test_labels, train_values, 100, printError=True)

################################ Part B #######################################################

	print("\n\n Part B")
	print("tree number, train error, test error")
	tree_numbers = [1,5,10,50,100,500,1000]
	for tree_num in tree_numbers:
		models = EnsembleLearning.Bagging(train_data, train_labels, train_values, 500, tree_num)
		train_error = EnsembleLearning.testBagging(models, train_data, train_labels)
		test_error = EnsembleLearning.testBagging(models, test_data, test_labels)
		print(tree_num, train_error, test_error)

################################# Part C #######################################################

	print("\n\nPart C")
	bagged_predictors = []
	num_runs = 100
	num_trees = 1000
	for run in range(num_runs):
		print(run)
		# get data subset
		data_subset = []
		labels_subset = []
		for m in range(1000):
			index = randint(0, len(train_data)-1)
			data_subset.append(train_data[index])
			labels_subset.append(train_labels[index])
		data_subset = np.array(data_subset)
		labels_subset = np.array(labels_subset)

		# get 1000 models
		models = EnsembleLearning.Bagging(data_subset, labels_subset, train_values, 300, num_trees)
		bagged_predictors.append(models)


	# get single tree bias and variance
	avg_preds = []
	variances = []
	for t_index in range(len(test_data)):
		example = test_data[t_index]
		label = test_labels[t_index]
		preds = []
		for index in range(num_runs):
			first_tree = bagged_predictors[index][0]
			prediction = DecisionTree.traceback(example, first_tree)
			if prediction == "yes":
				preds.append(1)
			else:
				preds.append(-1)
		example_avg = np.mean(np.array(preds))
		avg_preds.append(example_avg)
		variances.append(np.var(preds))

	bias = np.mean((avg_preds - int_labels)**2)
	variance = np.mean(variances)
	print("Single tree bias:")
	print(bias)
	print("Single tree variance:")
	print(variance)
	print("Single tree general squared error:")
	print(bias+variance)
	print()

	# get bagged predictor bias and variance
	avg_preds = []
	variances = []
	for t_index in range(len(test_data)):
		example = test_data[t_index]
		label = test_labels[t_index]
		preds = []
		for index in range(num_runs):
			trees = bagged_predictors[index]
			summation = 0
			for tree in trees:
				prediction = DecisionTree.traceback(example, tree)
				if prediction == "yes":
					summation += 1
				else:
					summation += -1
			if summation < 0:
				preds.append(-1)
			else:
				preds.append(1)
		example_avg = np.mean(np.array(preds))
		avg_preds.append(example_avg)
		variances.append(np.var(preds))

	bias = np.mean((avg_preds - int_labels)**2)
	variance = np.mean(variances)
	print("Bagged bias:")
	print(bias)
	print("Bagged variance:")
	print(variance)
	print("Bagged general squared error:")
	print(bias+variance)
	print()

################################# Part D #######################################################

	print("\n\n Part D")
	tree_numbers = [1,5,10,50,100,500,1000]
	attr_numbers = [2,4,6]
	for attr_num in attr_numbers:
		print(str(attr_num) + " attributes")
		for tree_num in tree_numbers:
			models = EnsembleLearning.RandomForest(train_data, train_labels, train_values, 500, attr_num, tree_num)
			train_error = EnsembleLearning.testBagging(models, train_data, train_labels)
			test_error = EnsembleLearning.testBagging(models, test_data, test_labels)
			print(tree_num, train_error, test_error)

################################# Part E #######################################################

	print("\n\n Part E")
	forest_predictors = []
	for run in range(100):
		print(run)
		# get data subset
		data_subset = []
		labels_subset = []
		for m in range(1000):
			index = randint(0, len(train_data)-1)
			data_subset.append(train_data[index])
			labels_subset.append(train_labels[index])
		data_subset = np.array(data_subset)
		labels_subset = np.array(labels_subset)

		# get 1000 models
		models = EnsembleLearning.RandomForest(data_subset, labels_subset, train_values, 300, 6, 1000)
		forest_predictors.append(models)

	# get single tree bias and variance
	avg_preds = []
	variances = []
	for t_index in range(len(test_data)):
		example = test_data[t_index]
		label = test_labels[t_index]
		preds = []
		for index in range(10):
			first_tree = forest_predictors[index][0]
			prediction = DecisionTree.traceback(example, first_tree)
			if prediction == "yes":
				preds.append(1)
			else:
				preds.append(-1)
		example_avg = np.mean(np.array(preds))
		avg_preds.append(example_avg)
		variances.append(np.var(preds))

	bias = np.mean((avg_preds - int_labels)**2)
	variance = np.mean(variances)
	print("Single tree bias:")
	print(bias)
	print("Single tree variance:")
	print(variance)
	print("Single tree general squared error:")
	print(bias+variance)
	print()

	# get bagged predictor bias and variance
	avg_preds = []
	variances = []
	for t_index in range(len(test_data)):
		example = test_data[t_index]
		label = test_labels[t_index]
		preds = []
		for index in range(10):
			trees = forest_predictors[index]
			summation = 0
			for tree in trees:
				prediction = DecisionTree.traceback(example, tree)
				if prediction == "yes":
					summation += 1
				else:
					summation += -1
			if summation < 0:
				preds.append(-1)
			else:
				preds.append(1)
		example_avg = np.mean(np.array(preds))
		avg_preds.append(example_avg)
		variances.append(np.var(preds))

	bias = np.mean((avg_preds - int_labels)**2)
	variance = np.mean(variances)
	print("Bagged bias:")
	print(bias)
	print("Bagged variance:")
	print(variance)
	print("Bagged general squared error:")
	print(bias+variance)
	print()







