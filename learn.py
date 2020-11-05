import numpy as np
from pyannow import math

def learn(inputs, targets, weights=[0.1,0.1], bias=1, lrate=0.05, nepoch=10000, lrule='delta', actfunc='threshold'):

    for epoch in range(nepoch):

        # feedforward inputs
        if bias is None:
            ins = np.dot(inputs, weights)
        else:
            ins = np.dot(inputs, weights) + bias

        # feedforward outputs
        if actfunc == 'sigmoid':
            outs = math.sigmoid(ins)
            derr = math.sigmoid_der(outs)
        elif actfunc == 'linear':
            outs = ins
            derr = np.ones_like(outs)
        else:
            outs = threshold(ins)
            derr = np.ones_like(outs)

        err = outs - targets
        derr_err = err * derr
        der_f = np.dot(inputs.T,derr_err)

        weights -= lrate * der_f

        if bias is not None:
            err_sum = derr_err.sum()
            bias -= lrate * err_sum
        
        # for i in derr_err:
        #     bias -= lrate * i
    
        tss = math.tss(outs, targets)
        if tss == 0:
            break

    if bias is not None:
        bias = bias.round(3)
    res = {'outputs':outs.round(3), 
    'weights':weights.round(3), 
    'bias':bias, 
    'tss':tss.round(3), 
    'epoch':epoch}

    return res

# def slff(inputs, targets, weights=[0.1], bias=[1], lrate=0.05, nepoch=1000, lrule='delta', actfunc='threshold'):

    
#     res = {'outputs':outs, 'weights':weights, 'bias':bias, 'derivatives':der_f, 'tss':tss}

#     return res

def threshold(x, th=0, up=1, down=0):
    return np.where(x>th, up, down)

def linear(x, th=0, up=1, down=0):
    return np.where(x>th, up, down)