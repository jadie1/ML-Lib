# Jadie Adams
import DecisionTree
import numpy as np
import sys
sys.path.append("..")
import DataUtils

if __name__ == "__main__":
	output = open("CarResults.csv", "w+")
	output.write("heuristic, depth, train accuracy, test accuracy \n")
	train_data, train_labels, test_data, test_labels = DataUtils.getData('../Data/car')
	train_values = [[0,np.array(['vhigh', 'high', 'med', 'low'])],
		[1,np.array(['vhigh', 'high', 'med', 'low'])],
		[2,np.array(['2', '3', '4', '5more'])],
		[3,np.array(['2', '4', 'more'])],
		[4,np.array(['small', 'med', 'big'])],
		[5,np.array(['low', 'med', 'high'])]]
	heuristics = ["information_gain", "gini_index", "majority_error"]
	for heuristic in heuristics:
		print(heuristic)
		for max_depth in range(1,7):
			tree = DecisionTree.ID3(train_data, train_labels, train_values, heuristic, 0, max_depth)
			train_accuracy = DecisionTree.testTree(tree, train_data, train_labels)
			test_accuracy = DecisionTree.testTree(tree, test_data, test_labels)
			print(str(max_depth) + " " + str(train_accuracy) + " " +  str(test_accuracy))
			output.write(heuristic + "," + str(max_depth) + "," + str(train_accuracy) + "," +  str(test_accuracy) + '\n')
	output.close()
			