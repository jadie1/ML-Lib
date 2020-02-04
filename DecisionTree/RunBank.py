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


if __name__ == "__main__":
	output = open("BankResults.csv", "w+")
	output.write("heuristic, depth, train accuracy, test accuracy \n")
	train_data, train_labels = getData("bank/train.csv")
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
	train_data = binarize(train_data, [0,5,9,11,12,13,14])
	test_data, test_labels = getData("bank/test.csv")
	test_data = binarize(test_data, [0,5,9,11,12,13,14])

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
	train_data, train_labels = replaceMissing(train_data, train_labels)
	test_data, test_labels = replaceMissing(test_data, test_labels)
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
			