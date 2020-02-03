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
	train_data, train_labels = getData("car/train.csv")
	test_data, test_labels = getData("car/test.csv")
	heuristic = "info_gain"
	max_depth = 100
	tree = DecisionTree.ID3(train_data, train_labels, heuristic, max_depth)