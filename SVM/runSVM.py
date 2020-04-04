#Jadie Adams
import os
import numpy as np 
import SVM
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
	epochs = 100
	Cs = [100/873, 500/873, 700/873]
	init = 0.00001
	d = 2


	# learning rate functions
	def lrf1(t):
		return init/(1+(init/d)*t)

	print("SVM in primal domain with stocahstic gradient descent.")
	learning_rate_function = "lr1"
	for C in Cs:
		print('\nC value = ' + str(C))
		parameters =SVM.PrimalSVM(epochs, lrf1, C, train_data, train_labels)
		print("Parameters:")
		print(parameters)
		error = SVM.test(parameters, train_data, train_labels)
		print("Train error: " + str(error))
		error = SVM.test(parameters, test_data, test_labels)
		print("Test error: " + str(error))