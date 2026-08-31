import torch
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
import torch.nn as nn

CONFIDENCE_THRESHOLD = 0.5

def accuracy(output, target):
    with torch.no_grad():
        m = nn.Sigmoid()
        output = m(output)
        output = output.flatten()
        output[output>CONFIDENCE_THRESHOLD] = 1
        output[output<=CONFIDENCE_THRESHOLD] = 0
        if len(output.shape) > 1:
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def recall(output, target):
    m = nn.Sigmoid()
    output = m(output)
    output = output.flatten()
    output[output > CONFIDENCE_THRESHOLD] = 1
    output[output <= CONFIDENCE_THRESHOLD] = 0
    with torch.no_grad():
        if len(output.shape) > 1:
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
        assert pred.shape[0] == len(target)

        r = recall_score(target.cpu(), pred.cpu(), average='binary', zero_division=0)
    return r


def precision(output, target):
    m = nn.Sigmoid()
    output = m(output)
    output = output.flatten()
    output[output > CONFIDENCE_THRESHOLD] = 1
    output[output <= CONFIDENCE_THRESHOLD] = 0
    with torch.no_grad():

        if len(output.shape) > 1:
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
        assert pred.shape[0] == len(target)

        p = precision_score(target.cpu(), pred.cpu(), average='binary', zero_division=0)
    return p


def f1(output, target):
    m = nn.Sigmoid()
    output = m(output)
    output = output.flatten()
    output[output > CONFIDENCE_THRESHOLD] = 1
    output[output <= CONFIDENCE_THRESHOLD] = 0
    with torch.no_grad():

        if len(output.shape) > 1:
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
        assert pred.shape[0] == len(target)


        f = f1_score(target.cpu(), pred.cpu(), average='binary', zero_division=0)
    return f


# def top_k_acc(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)

#
# def metric_helper(output, target):
#     with torch.no_grad():
#         output_bin=((output[:,1].reshape(-1))>0.5)*1
#         target_bin=((target.reshape(-1))>0.5)*1
#         #error count
#         err=(torch.logical_xor(output_bin, target_bin)*1).sum()
#         #false negative count
#         falseNeg=np.array([1 for i in range(target_bin.shape[0]) if target_bin[i]==1 and output_bin[i]==0]).sum()
#         #total predictions
#         total=target_bin.shape[0]
#         #total cases labeled positive
#         truth=target_bin.sum()
#     return total, err, falseNeg, truth
#
# def precision(output, target):
#     total, err, falseNeg, truth=metric_helper(output, target)
#     falsePos=err-falseNeg
#     return (truth-falseNeg)/(truth-falseNeg+falsePos)*100 if (truth-falseNeg+falsePos) != 0 else 0
#
# def recall(output, target):
#     total, err, falseNeg, truth=metric_helper(output, target)
#     falsePos=err-falseNeg
#     # print(total, err, falseNeg, truth)
#     return (truth-falseNeg)/(truth)*100 if truth != 0 else 0
#
#
# def accuracy(output, target):
#     total, err, falseNeg, truth=metric_helper(output, target)
#     return (total-err)/total*100
#
# def f1(output, target):
#     p=precision(output, target)
#     r=recall(output, target)
#     return 2*p*r/(p+r) if (p+r) != 0 else 0
#
#
# def actual_positive(output, target):
#     return len([1 for i in target if i>0.5])/1.0
#
# def actual_negative(output, target):
#     return len([1 for i in target if i<0.5])/1.0