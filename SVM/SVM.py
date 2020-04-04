# JAdie Adams
import os 
import numpy as np
from sklearn.utils import shuffle

reg_strength = 10000

def predict(w, x):
    return np.sign(np.dot(w,x))

def test(w, data, labels):
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    wrong = 0
    for index in range(inputs.shape[0]):
        if labels[index] != predict(w, inputs[index]):
            wrong += 1
    return((wrong/inputs.shape[0])*100)

def update(C, lr, n, w, x, y):  
    val = (y * np.dot(x, w))
    if val <= 1:
        w = w - lr*w + lr*C*n*y*x
    else:
        w = (1-lr)*w
    return w

def PrimalSVM(epochs, lr_function, C, data, labels):
    print_train = False
    w = np.full((data.shape[1] + 1), 0)
    inputs = np.ones((data.shape[0], data.shape[1]+1))
    inputs[:,:-1] = data
    n = inputs.shape[0]
    if print_train:
        print(test(w, data, labels))
    for e in range(epochs):
        learning_rate = lr_function(e)
        X, Y = shuffle(inputs, labels)
        for index in range(n):
            x = X[index]
            y = Y[index]
            w = update(C, learning_rate, n, w, x, y)
    return w