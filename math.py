
import numpy as np

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def tss(x, y):
    err = y - x
    err_sq = err**2
    return err_sq.sum()