# Jadie Adams

import os
import numpy as np
import LinearRegression
import sys
sys.path.append("..")
import DataUtils

# w∗= (XX^T)−1(XY)
def getLinearSolution(data, labels):
	weights = []
	x_transpose = np.ones((data.shape[0], data.shape[1]+1))
	x_transpose[:,:-1] = data
	x = np.transpose(x_transpose)
	y = labels
	part0 = np.dot(x,x_transpose)
	part1 = np.linalg.inv(part0)
	part2 = np.dot(x,y)
	weights = np.dot(part1,part2)
	return weights


if __name__ == "__main__":
	# Get args
	dataFolder = sys.argv[1]

	# Get Data
	train_data, train_labels, test_data, test_labels = DataUtils.getData(dataFolder)

	all_data = np.concatenate((train_data, test_data), axis=0)
	all_labels = np.concatenate((train_labels, test_labels), axis=0).astype('float64')

	# all_data = np.array([[1,-1,2],[1,1,3],[-1,1,0],[1,2,-4],[3,-1,-1]])
	# all_labels = np.array([[1],[4],[-1],[-2],[0]])

	# Get analytic linear solution
	weights = getLinearSolution(all_data, all_labels)
	print("Optimal weights:")
	print(weights)

	print("Error on test data:")
	error = LinearRegression.test(weights,test_data,test_labels)
	print(error)

	