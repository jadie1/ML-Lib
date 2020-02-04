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
	# train_data, train_labels = getData("car/train.csv")
	# train_values = [np.array(['vhigh', 'high', 'med', 'low']),
	# 	np.array(['vhigh', 'high', 'med', 'low']),
	# 	np.array(['2', '3', '4', '5more']),
	# 	np.array(['2', '4', 'more']),
	# 	np.array(['small', 'med', 'big']),
	# 	np.array(['low', 'med', 'high'])]
	train_data, train_labels = getData("try.csv")
	train_values = [np.array(['S',"O","R"]),
		np.array(['H', "O", "C"]),
		np.array(["H","N", "L"]),
		np.array(["S", "w"])]
	heuristic = "information_gain"
	# heuristic = "gini_index"
	# heuristic = "majority_error"
	max_depth = 1
	tree = DecisionTree.ID3(train_data, train_labels, train_values, heuristic, 0, max_depth)
	print(tree)
	test_data, test_labels = getData("car/test.csv")