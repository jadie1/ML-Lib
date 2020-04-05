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
	d = 10


	# learning rate functions
	def lrf1(t):
		return init/(1+((init/d)*t))

	def lrf2(t):
		return init/(1*t)

	# Question 2.2
	print("SVM in primal domain with stocahstic gradient descent.")
	print("\nWith first learning rate function: ")
	for C in Cs:
		print('\nC value = ' + str(C))
		parameters =SVM.PrimalSVM(epochs, lrf1, C, train_data, train_labels)
		print("Parameters:")
		print(parameters)
		error = SVM.test(parameters, train_data, train_labels)
		print("Train error: " + str(error))
		error = SVM.test(parameters, test_data, test_labels)
		print("Test error: " + str(error))
	input("\nWith second learning rate function: ")
	for C in Cs:
		print('\nC value = ' + str(C))
		parameters =SVM.PrimalSVM(epochs, lrf2, C, train_data, train_labels)
		print("Parameters:")
		print(parameters)
		error = SVM.test(parameters, train_data, train_labels)
		print("Train error: " + str(error))
		error = SVM.test(parameters, test_data, test_labels)
		print("Test error: " + str(error))


	# kernels
	def normal_kernel(x,y):
		return np.dot(x,y)

	def gaussian_kernel(x, y, gamma):
		return np.exp(-np.sum(np.square(x - y)) / gamma)

	# # Question 2.3.A
	print("\n#####SVM in dual domain.######")
	kernel = lambda x, y: normal_kernel(x, y)
	for C in Cs:
		print('\nC value = ' + str(C))
		parameters = SVM.DualSVM(kernel, C, train_data, train_labels, test_data, test_labels)
		print(parameters)
		# error = SVM.test(parameters, train_data, train_labels)
		# print("Train error: " + str(error))
		# error = SVM.test(parameters, test_data, test_labels)
		# print("Test error: " + str(error))

	# Question 2.3.B
	print("\n ####SVM in dual domain with Gaussian kernel#######")
	gammas = [0.1,0.5,1,5,100]
	for C in Cs:
		print('\nC value = ' + str(C))
		for gamma in gammas:
			print('Gamma value = ' +str(gamma))
			kernel = lambda x, y: gaussian_kernel(x, y, gamma)
			parameters = SVM.DualSVM(kernel, C, train_data, train_labels, test_data, test_labels)
			print(parameters)
			
