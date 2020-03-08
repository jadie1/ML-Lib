import os
import numpy as np 
import Perceptron
import sys
sys.path.append("..")
import DataUtils

if __name__ == "__main__":
	# Get args
	dataFolder = "../Data/bank-note/"

	# Get Data
	train_data, train_labels, test_data, test_labels = DataUtils.getData(dataFolder)
	train_labels = DataUtils.fixLabels(train_labels)
	test_labels = DataUtils.fixLabels(test_labels)
	learning_rate = 0.1
	epochs = 10

	print("Standard Perceptron:")
	weights = Perceptron.standard_perceptron(epochs, learning_rate, train_data, train_labels)
	print("Weights:")
	print(weights)
	error = Perceptron.test(weights, train_data, train_labels)
	print("Train error: " + str(error))
	error = Perceptron.test(weights, test_data, test_labels)
	print("Test error: " + str(error))

	print("\nVoted Perceptron:")
	W, C = Perceptron.voted_perceptron(epochs, learning_rate, train_data, train_labels)
	error = Perceptron.test_voted(W, C, train_data, train_labels)
	# print("Weight vector, Count")
	# for index in range(len(W)):
	# 	print(str(W[index]) + " & " + str(C[index]) + " \\\\" )
	print("Train error: " + str(error))
	error = Perceptron.test_voted(W, C, test_data, test_labels)
	print("Test error: " + str(error))

	print("\nAverage Perceptron:")
	weights = Perceptron.average_perceptron(epochs, learning_rate, train_data, train_labels)
	print("Weights:")
	print(weights)
	error = Perceptron.test(weights, train_data, train_labels)
	print("Train error: " + str(error))
	error = Perceptron.test(weights, test_data, test_labels)
	print("Test error: " + str(error))


