Jadie Adams
u0930409
Machine Learning HW 1

The code is located at: https://github.com/jadie1/ML-Lib/tree/master/DecisionTree

To see examples run:
	chmod +x run.sh
	./run.sh

This calls RunCar.py and RunBank.py
These scripts print the results and save them to output files 'CarResults.csv' and 'BankResults.csv'
Note running these scripts requires the car and bank data folders to be in the directory
The decision tree implementation code is all in DecisionTree.py

To make a new decision tree:
	1. Add data to Data folder
	2. Make a Run.py which:
		getData() and other needed data proccessing steps from DataUtils.py
		initializes a list of all possible attribute values
		builds a tree by calling:  DecisionTree.ID3(train_data, train_labels, train_values, heuristic, 0, max_depth)
		tests the tree by calling: DecisionTree.testTree(tree, test_data, test_labels)