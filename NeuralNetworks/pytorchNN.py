# Jadie Adams
# Reference: https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import os
import sys
import torch.optim as optim
sys.path.append("..")
import DataUtils

class Net3(nn.Module):
	def __init__(self, input_dim, hid_dim, activation):
		super(Net3, self).__init__()
		# Number of input features is 12.
		self.layer_1 = nn.Linear(input_dim, hid_dim) 
		self.layer_2 = nn.Linear(hid_dim, hid_dim)
		self.layer_out = nn.Linear(hid_dim, 1) 
		if activation == 'ReLU':
			self.activation = nn.ReLU()
			torch.nn.init.kaiming_normal_(self.layer_1.weight)
			torch.nn.init.kaiming_normal_(self.layer_2.weight)
			torch.nn.init.kaiming_normal_(self.layer_out.weight)
		elif activation == "tanh":
			self.activation = nn.Tanh()
			torch.nn.init.xavier_uniform_(self.layer_1.weight)
			torch.nn.init.xavier_uniform_(self.layer_2.weight)
			torch.nn.init.xavier_uniform_(self.layer_out.weight)
		else:
			print("error activation unimplemented")
		
	def forward(self, inputs):
		x = self.activation(self.layer_1(inputs))
		x = self.activation(self.layer_2(x))
		x = self.layer_out(x)
		return x

class Net5(nn.Module):
	def __init__(self, input_dim, hid_dim, activation):
		super(Net5, self).__init__()
		# Number of input features is 12.
		self.layer_1 = nn.Linear(input_dim, hid_dim) 
		self.layer_2 = nn.Linear(hid_dim, hid_dim)
		self.layer_3 = nn.Linear(hid_dim, hid_dim)
		self.layer_4 = nn.Linear(hid_dim, hid_dim)
		self.layer_out = nn.Linear(hid_dim, 1) 
		if activation == 'ReLU':
			self.activation = nn.ReLU()
			torch.nn.init.kaiming_normal_(self.layer_1.weight)
			torch.nn.init.kaiming_normal_(self.layer_2.weight)
			torch.nn.init.kaiming_normal_(self.layer_3.weight)
			torch.nn.init.kaiming_normal_(self.layer_4.weight)
			torch.nn.init.kaiming_normal_(self.layer_out.weight)
		elif activation == "tanh":
			self.activation = nn.Tanh()
			torch.nn.init.xavier_uniform_(self.layer_1.weight)
			torch.nn.init.xavier_uniform_(self.layer_2.weight)
			torch.nn.init.xavier_uniform_(self.layer_3.weight)
			torch.nn.init.xavier_uniform_(self.layer_4.weight)
			torch.nn.init.xavier_uniform_(self.layer_out.weight)
		else:
			print("error activation unimplemented")
		
	def forward(self, inputs):
		x = self.activation(self.layer_1(inputs))
		x = self.activation(self.layer_2(x))
		x = self.activation(self.layer_3(x))
		x = self.activation(self.layer_4(x))
		x = self.layer_out(x)
		return x

class Net7(nn.Module):
	def __init__(self, input_dim, hid_dim, activation):
		super(Net7, self).__init__()
		# Number of input features is 12.
		self.layer_1 = nn.Linear(input_dim, hid_dim) 
		self.layer_2 = nn.Linear(hid_dim, hid_dim)
		self.layer_3 = nn.Linear(hid_dim, hid_dim)
		self.layer_4 = nn.Linear(hid_dim, hid_dim)
		self.layer_5 = nn.Linear(hid_dim, hid_dim)
		self.layer_6 = nn.Linear(hid_dim, hid_dim)
		self.layer_out = nn.Linear(hid_dim, 1) 
		if activation == 'ReLU':
			self.activation = nn.ReLU()
			torch.nn.init.kaiming_normal_(self.layer_1.weight)
			torch.nn.init.kaiming_normal_(self.layer_2.weight)
			torch.nn.init.kaiming_normal_(self.layer_3.weight)
			torch.nn.init.kaiming_normal_(self.layer_4.weight)
			torch.nn.init.kaiming_normal_(self.layer_5.weight)
			torch.nn.init.kaiming_normal_(self.layer_6.weight)
			torch.nn.init.kaiming_normal_(self.layer_out.weight)
		elif activation == "tanh":
			self.activation = nn.Tanh()
			torch.nn.init.xavier_uniform_(self.layer_1.weight)
			torch.nn.init.xavier_uniform_(self.layer_2.weight)
			torch.nn.init.xavier_uniform_(self.layer_3.weight)
			torch.nn.init.xavier_uniform_(self.layer_4.weight)
			torch.nn.init.xavier_uniform_(self.layer_5.weight)
			torch.nn.init.xavier_uniform_(self.layer_6.weight)
			torch.nn.init.xavier_uniform_(self.layer_out.weight)
		else:
			print("error activation unimplemented")
		
	def forward(self, inputs):
		x = self.activation(self.layer_1(inputs))
		x = self.activation(self.layer_2(x))
		x = self.activation(self.layer_3(x))
		x = self.activation(self.layer_4(x))
		x = self.activation(self.layer_5(x))
		x = self.activation(self.layer_6(x))
		x = self.layer_out(x)
		return x

class MyTrainSet():
	def __init__(self, train_data, train_labels):
		self.data = np.array(train_data) 
		self.targets = train_labels.reshape((train_labels.shape[0],1))
	def __getitem__(self, index):
		return torch.FloatTensor(self.data[index]),torch.FloatTensor(self.targets[index])
	def __len__(self):
		return len(self.data)

class MyTestSet():
	def __init__(self, test_data):
		self.data = np.array(test_data)
	def __getitem__(self, index):
		return torch.FloatTensor(self.data[index])
	def __len__(self):
		return len(self.data)

def binary_acc(y_pred, y_test):
	y_pred_tag = torch.round(torch.sigmoid(y_pred))
	correct_results_sum = (y_pred_tag == y_test).sum().float()
	acc = correct_results_sum/y_test.shape[0]
	acc = torch.round(acc * 100)
	return acc

def TrainAndTest(model, epochs, learning_rate, train_loader, test_loader):
	model.to(device)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	model.train()
	epoch_acc = 0
	for e in range(1,epochs+1):
		if epoch_acc/len(train_loader) == 100:
			break
		epoch_loss = 0
		epoch_acc = 0
		for X_batch, y_batch in train_loader:
			X_batch, y_batch = X_batch.to(device), y_batch.to(device)
			optimizer.zero_grad()
			y_pred = model(X_batch)
			loss = criterion(y_pred, y_batch)
			acc = binary_acc(y_pred, y_batch)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			epoch_acc += acc.item()
		print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

	y_pred_list = []
	model.eval()
	with torch.no_grad():
		for X_batch in test_loader:
			X_batch = X_batch.to(device)
			y_test_pred = model(X_batch)
			y_test_pred = torch.sigmoid(y_test_pred)
			y_pred_tag = torch.round(y_test_pred)
			y_pred_list.append(y_pred_tag.cpu().numpy())

	y_pred_list = [a.squeeze().tolist() for a in y_pred_list][0]
	
	errors = 0
	for index in range(test_labels.shape[0]):
		if y_pred_list[index] != test_labels[index]:
			errors += 1
	print("Test Accuracy: " + str((1- errors/len(y_pred_list))*100))

if __name__ == "__main__":
	# Please try two activation functions, ``tanh'' and ``RELU''.  For ``tanh", 
	# please use the ``Xavier' initialization; and for ``RELU'', 
	# please use the ``he'' initialization. 
	# Vary the depth from $\{3, 5, 9\} $ and width from $\{5, 10, 25, 50, 100\}$. 
	# Please use the Adam optimizer for training.

	dataFolder = "../Data/bank-note/"

	# Get Data
	train_data, train_labels, test_data, test_labels = DataUtils.getData(dataFolder)
	train_data = train_data.astype('float64')
	train_labels = train_labels.astype('float64')
	test_data = test_data.astype("float64")
	test_labels = test_labels.astype('float64')

	# get customized torch datasets
	train_dataset = MyTrainSet(train_data, train_labels)
	train_loader = DataLoader(
		train_dataset,
		batch_size=132,
		shuffle=True,
		num_workers=2,
		pin_memory=torch.cuda.is_available()
	)
	test_dataset = MyTestSet(test_data)
	test_loader = DataLoader(
		test_dataset,
		batch_size=test_data.shape[0],
		shuffle=False,
		num_workers=2,
		pin_memory=torch.cuda.is_available()
	)

	# Set Parameters
	learning_rate = 0.01
	epochs = 100
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	hidden_dims = [5,10,25,50,100]

	print("###### ReLU activation and He initialization ######")
	print("\n#### Network Depth = 3 ####")
	for hid_dim in hidden_dims:
		print("## Network Width = " + str(hid_dim) + ' ##')
		model = Net3(train_data.shape[1], hid_dim, 'ReLU')
		TrainAndTest(model, epochs, learning_rate, train_loader, test_loader)
	print("\n#### Network Depth = 5 ####")
	for hid_dim in hidden_dims:
		print("## Network Width = " + str(hid_dim) + ' ##')
		model = Net5(train_data.shape[1], hid_dim, 'ReLU')
		TrainAndTest(model, epochs, learning_rate, train_loader, test_loader)
	print("\n#### Network Depth = 7 ####")
	for hid_dim in hidden_dims:
		print("## Network Width = " + str(hid_dim) + ' ##')
		model = Net7(train_data.shape[1], hid_dim, 'ReLU')
		TrainAndTest(model, epochs, learning_rate, train_loader, test_loader)
	
	print("\n\n###### tanh activation and Xavier initialization ######")
	print("\n#### Network Depth = 3 ####")
	for hid_dim in hidden_dims:
		print("## Network Width = " + str(hid_dim) + ' ##')
		model = Net3(train_data.shape[1], hid_dim, 'tanh')
		TrainAndTest(model, epochs, learning_rate, train_loader, test_loader)
	print("\n#### Network Depth = 5 ####")
	for hid_dim in hidden_dims:
		print("## Network Width = " + str(hid_dim) + ' ##')
		model = Net5(train_data.shape[1], hid_dim, 'tanh')
		TrainAndTest(model, epochs, learning_rate, train_loader, test_loader)
	print("\n#### Network Depth = 7 ####")
	for hid_dim in hidden_dims:
		print("## Network Width = " + str(hid_dim) + ' ##')
		model = Net7(train_data.shape[1], hid_dim, 'tanh')
		TrainAndTest(model, epochs, learning_rate, train_loader, test_loader)
