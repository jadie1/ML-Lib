# Jadie Adams
import DecisionTree
import numpy as np
import sys
sys.path.append("..")
import DataUtils

if __name__ == "__main__":
	output = open("BankResults.csv", "w+")
	output.write("heuristic, depth, train accuracy, test accuracy \n")
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

	print("Results when 'unknown' is an attribute.")
	heuristics = ["information_gain", "majority_error", "gini_index"]
	for heuristic in heuristics:
		print(heuristic)
		for max_depth in range(1,17):
			tree = DecisionTree.ID3(train_data, train_labels, train_values, heuristic, 0, max_depth)
			train_accuracy = DecisionTree.testTree(tree, train_data, train_labels)
			test_accuracy = DecisionTree.testTree(tree, test_data, test_labels)
			print(str(max_depth) +"  "+str(train_accuracy) + "  " +  str(test_accuracy))
			output.write(heuristic + "," + str(max_depth) + "," + str(train_accuracy) + "," +  str(test_accuracy) + '\n')

	output.write("\nheuristic, depth, train accuracy, test accuracy \n")
	print("\nResults when 'unknown' is a missing value.")
	train_data, train_labels = DataUtils.replaceMissing(train_data, train_labels)
	test_data, test_labels = DataUtils.replaceMissing(test_data, test_labels)
	heuristics = ["information_gain", "majority_error", "gini_index"]
	for heuristic in heuristics:
		print(heuristic)
		for max_depth in range(1,17):
			tree = DecisionTree.ID3(train_data, train_labels, train_values, heuristic, 0, max_depth)
			train_accuracy = DecisionTree.testTree(tree, train_data, train_labels)
			test_accuracy = DecisionTree.testTree(tree, test_data, test_labels)
			print(str(max_depth) +" "+str(train_accuracy) + " " +  str(test_accuracy))
			output.write(heuristic + "," + str(max_depth) + "," + str(train_accuracy) + "," +  str(test_accuracy) + '\n')
	output.close()
			