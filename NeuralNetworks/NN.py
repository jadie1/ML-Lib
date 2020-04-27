# Jadie Adams
# Reference: https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
import numpy as np 

def get_archtecture(x_dim, hid_dim):
	arch = [
	{"input_dim": x_dim, "output_dim": hid_dim},
	{"input_dim": hid_dim, "output_dim": hid_dim},
	{"input_dim": hid_dim, "output_dim": 1}]
	return arch

def initialize(arch, seed):
	np.random.seed(seed)
	number_of_layers = len(arch)
	params_values = {}
	for idx, layer in enumerate(arch):
		layer_idx = idx+ 1
		layer_input_size = layer["input_dim"]
		layer_output_size = layer["output_dim"]
		params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
		params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1
	return params_values

def zero_initialize(arch, seed):
	np.random.seed(seed)
	number_of_layers = len(arch)
	params_values = {}
	for idx, layer in enumerate(arch):
		layer_idx = idx+ 1
		layer_input_size = layer["input_dim"]
		layer_output_size = layer["output_dim"]
		params_values['W' + str(layer_idx)] = np.zeros((layer_output_size, layer_input_size))
		params_values['b' + str(layer_idx)] = np.zeros((layer_output_size, 1))
	return params_values

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def sigmoid_backward(dA, Z):
	sig = sigmoid(Z)
	return dA * sig * (1 - sig)

def shuffle(x, y):
	indices = np.arange(x.shape[1])
	np.random.shuffle(indices)
	x = x[:,indices]
	y = y[:,indices]
	return x,y

def layer_forward(A_prev, W_curr, b_curr):
	Z_curr = np.dot(W_curr, A_prev) + b_curr
	return sigmoid(Z_curr), Z_curr

def forward_propagation(X, params_values, arch):
	memory = {}
	A_curr = X
	for idx, layer in enumerate(arch):
		layer_idx = idx + 1
		A_prev = A_curr
		W_curr = params_values["W" + str(layer_idx)]
		b_curr = params_values["b" + str(layer_idx)]
		A_curr, Z_curr = layer_forward(A_prev, W_curr, b_curr)
		memory["A" + str(idx)] = A_prev
		memory["Z" + str(layer_idx)] = Z_curr
	return A_curr, memory

def layer_back(dA_curr, W_curr, b_curr, Z_curr, A_prev):
	m = A_prev.shape[1]
	dZ_curr = sigmoid_backward(dA_curr, Z_curr)
	dW_curr = np.dot(dZ_curr, A_prev.T) / m
	db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
	dA_prev = np.dot(W_curr.T, dZ_curr)
	return dA_prev, dW_curr, db_curr

def back_propogation(Y_hat, Y, memory, params_values, arch):
	grads_values = {}
	m = Y.shape[1]
	Y = Y.reshape(Y_hat.shape)
	dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
	for layer_idx_prev, layer in reversed(list(enumerate(arch))):
		layer_idx_curr = layer_idx_prev + 1		
		dA_curr = dA_prev
		A_prev = memory["A" + str(layer_idx_prev)]
		Z_curr = memory["Z" + str(layer_idx_curr)]
		W_curr = params_values["W" + str(layer_idx_curr)]
		b_curr = params_values["b" + str(layer_idx_curr)]
		dA_prev, dW_curr, db_curr = layer_back(
			dA_curr, W_curr, b_curr, Z_curr, A_prev)
		grads_values["dW" + str(layer_idx_curr)] = dW_curr
		grads_values["db" + str(layer_idx_curr)] = db_curr
	return grads_values

def get_label(probs):
	probs_ = np.copy(probs)
	probs_[probs_ > 0.5] = 1
	probs_[probs_ <= 0.5] = 0
	return probs_

def get_accuracy(Y_hat, Y):
	Y_hat_ = get_label(Y_hat)
	return (Y_hat_ == Y).all(axis=0).mean()

def get_cost_value(Y_hat, Y):
	m = Y_hat.shape[1]
	cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
	return np.squeeze(cost)

def update(params_values, grads_values, arch, learning_rate):
	for layer_idx, layer in enumerate(arch):
		layer_idx = layer_idx+ 1
		params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]		
		params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
	return params_values;

def train(hid_dim, epochs, lr_func, data, labels):
	X = np.transpose(data.astype('float64'))
	Y = np.transpose(labels.reshape((labels.shape[0], 1)).astype('float64'))
	arch = get_archtecture(data.shape[1], hid_dim)
	init_seed = 2
	params_values = zero_initialize(arch, init_seed)
	cost_history = []
	accuracy_history = []
	for t in range(epochs):
		learning_rate = lr_func(t)
		X,Y = shuffle(X,Y)
		for i in range(X.shape[1]):
			x = X[:,i]
			x = x.reshape((x.shape[0], 1))
			y = Y[:,i]
			y = y.reshape((y.shape[0], 1))
			Y_hat, cache = forward_propagation(x, params_values, arch)
			cost = get_cost_value(Y_hat, y)
			cost_history.append(cost)
			accuracy = get_accuracy(Y_hat, y)
			accuracy_history.append(accuracy)
			grads_values = back_propogation(Y_hat, y, cache, params_values, arch)
			params_values = update(params_values, grads_values, arch, learning_rate)	
	return params_values

def test(hid_dim, data, labels, params_values):
	data = data.astype('float64')
	labels = labels.astype('float64')
	arch = get_archtecture(data.shape[1], hid_dim)
	Y_test_hat, _ = forward_propagation(np.transpose(data), params_values, arch)
	acc_test = get_accuracy(Y_test_hat, np.transpose(labels.reshape((labels.shape[0], 1))))
	return acc_test