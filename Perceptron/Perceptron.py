import os 
import numpy as np 

def predict(w, x):
    return np.sign(np.dot(w,x)+1e-10)

def standard_perceptron(epochs, r, data, labels, print_train=False):
    w = np.full((data.shape[1] + 1), 1e-5)
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    if print_train:
        print(test(w, data, labels))
    for e in range(epochs):
        for index in range(inputs.shape[0]):
            x = inputs[index]
            y = labels[index]
            if predict(w,x) != y:
                w = w + r*y*x
        if print_train:
            print(test(w, data, labels))
    return w

def test(w, data, labels):
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    wrong = 0
    for index in range(inputs.shape[0]):
        if labels[index] != predict(w, inputs[index]):
            wrong += 1
    return(wrong/inputs.shape[0])

def voted_perceptron(epochs, r, data, labels, print_train=False):
    w = np.full((data.shape[1] + 1), 1e-5)
    m = 0
    C = []
    W = []
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    for e in range(epochs):
        for index in range(inputs.shape[0]):
            x = inputs[index]
            y = labels[index]
            if predict(w,x) != y:
                w = w + r*y*x
                W.append(w)
                C.append(1)
            else:
                C[-1] += 1
    return W, C

def test_voted(W, C, data, labels):
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    wrong = 0
    for index in range(inputs.shape[0]):
        summation = 0
        for v_index in range(len(W)):
            summation += C[v_index]*predict(W[v_index], inputs[index])
        prediction = np.sign(summation)
        if labels[index] != prediction:
            wrong += 1
    return(wrong/inputs.shape[0])

def average_perceptron(epochs, r, data, labels, print_train=False):
    w = np.full((data.shape[1] + 1), 0)
    a = np.full((data.shape[1] + 1), 0)
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    if print_train:
        print(test(w, data, labels))
    for e in range(epochs):
        for index in range(inputs.shape[0]):
            x = inputs[index]
            y = labels[index]
            if predict(w,x) != y:
                w = w + r*y*x
            a = a + w
        if print_train:
            print(test(w, data, labels))
    return a