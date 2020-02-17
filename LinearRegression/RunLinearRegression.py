import os
import numpy as np 
import LinearRegression
import sys
sys.path.append("..")
import DataUtils

if __name__ == "__main__":
	# Get args
	dataFolder = sys.argv[1]
	gradient_descent_type = sys.argv[2]
	learning_rate = sys.argv[3]

	# Get Data
	train_data, train_labels, test_data, test_labels = DataUtils.getData(dataFolder)

	# Run stochastic gradient descent
	print("Training error:")
	if gradient_descent_type == 'stochastic':
		weights = LinearRegression.stochasticGradientDescent(learning_rate, train_data, train_labels)
	elif gradient_descent_type == 'batch':
		weights = LinearRegression.batchGradientDescent(learning_rate, train_data, train_labels)
	else:
		print("Error gradient descent type not recognized.")

	# Get test error
	print("\n\n Testing error:")
	test_error = LinearRegression.getError(weights, test_data, test_labels)
	print(test_error)
