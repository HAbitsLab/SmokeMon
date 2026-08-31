import torch
import torch
import numpy as np

def metric_helper(output, target):
    with torch.no_grad():
        output_bin=((output.reshape(-1))>0.5)*1
        target_bin=((target.reshape(-1))>0.5)*1
        #error count
        err=(torch.logical_xor(output_bin, target_bin)*1).sum()
        #false negative count
        falseNeg=np.array([1 for i in range(target_bin.shape[0]) if target_bin[i]==1 and output_bin[i]==0]).sum()
        #total predictions
        total=target_bin.shape[0]
        #total cases labeled positive
        truth=target_bin.sum()
    return total, err, falseNeg, truth

def precision(output, target):
    total, err, falseNeg, truth=metric_helper(output, target)
    falsePos=err-falseNeg
    return (truth-falseNeg)/(truth-falseNeg+falsePos)*100 if (truth-falseNeg+falsePos) != 0 else 0

def recall(output, target):
    total, err, falseNeg, truth=metric_helper(output, target)
    falsePos=err-falseNeg
    # print(total, err, falseNeg, truth)
    return (truth-falseNeg)/(truth)*100 if truth != 0 else 0


def accuracy(output, target):
    total, err, falseNeg, truth=metric_helper(output, target)
    return (total-err)/total*100

def f1(output, target):
    p=precision(output, target)
    r=recall(output, target)
    return 2*p*r/(p+r) if (p+r) != 0 else 0


def actual_positive(output, target):
    return len([1 for i in target if i>0.5])/1.0

def actual_negative(output, target):
    return len([1 for i in target if i<0.5])/1.0