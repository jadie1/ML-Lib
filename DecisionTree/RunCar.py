# Jadie Adams
import DecisionTree
import numpy as np 

# returns np array of arttributes and labels
def getData(csvFile):
	data = []
	with open(csvFile, 'r') as f:
		for line in f:
			data.append(line.strip().split(','))
	data = np.array(data)
	return data[:,:-1], data[:,-1]

if __name__ == "__main__":
	# train_data, train_labels = getData("try.csv")
	# train_values = [[0, np.array(['S',"O","R"])],
	# 	[1, np.array(['H', "O", "C"])],
	# 	[2, np.array(["H","N", "L"])],
	# 	[3, np.array(["S", "w"])]]
	# heuristic = "information_gain"
	# max_depth = 3
	# tree = DecisionTree.ID3(train_data, train_labels, train_values, heuristic, 0, max_depth)
	# test_data = train_data
	# test_labels = train_labels
	# accuracy = DecisionTree.testTree(tree, test_data, test_labels)

	output = open("CarResults.csv", "w+")
	output.write("heuristic, depth, train accuracy, test accuracy \n")
	train_data, train_labels = getData("car/train.csv")
	train_values = [[0,np.array(['vhigh', 'high', 'med', 'low'])],
		[1,np.array(['vhigh', 'high', 'med', 'low'])],
		[2,np.array(['2', '3', '4', '5more'])],
		[3,np.array(['2', '4', 'more'])],
		[4,np.array(['small', 'med', 'big'])],
		[5,np.array(['low', 'med', 'high'])]]
	test_data, test_labels = getData("car/test.csv")
	heuristics = ["information_gain", "gini_index", "majority_error"]
	max_depths = [1,2,3,4,5,6]
	for heuristic in heuristics:
		print(heuristic)
		for max_depth in max_depths:
			print(max_depth)
			tree = DecisionTree.ID3(train_data, train_labels, train_values, heuristic, 0, max_depth)
			train_accuracy = DecisionTree.testTree(tree, train_data, train_labels)
			test_accuracy = DecisionTree.testTree(tree, test_data, test_labels)
			print(str(train_accuracy) + " " +  str(test_accuracy))
			output.write(heuristic + "," + str(max_depth) + "," + str(train_accuracy) + "," +  str(test_accuracy) + '\n')
	output.close()
			