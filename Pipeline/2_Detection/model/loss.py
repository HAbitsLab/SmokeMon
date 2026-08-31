import torch
import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):

    return F.nll_loss(output, target)

def nll_loss_weighted(output, target):
    loss_weights = compute_loss_weights(output, target)
    return F.nll_loss(output, target, weight=loss_weights)

def binary_cross_entropy_with_logits(output, target):
    target = target.unsqueeze(1)
    target = target.float()
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    return loss(m(output), target)


def BCE_OHEM(x, y, ratio = 0.8):
    num_inst = x.size(0)
    num_hns = int(ratio * num_inst)
    x_ = x.clone()
    inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
    for idx, label in enumerate(y.data):
        inst_losses[idx] = -x_.data[idx]
        # loss_incs = -x_.sum(1)
    _, idxs = inst_losses.topk(num_hns)


    x_hn = x.index_select(0, idxs)
    y_hn = y.index_select(0, idxs)
    y_hn = y_hn.unsqueeze(1)
    y_hn = y_hn.float()

    m = nn.Sigmoid()
    loss = nn.BCELoss()
    return loss(m(x_hn), y_hn)


# class NLL_OHEM(th.nn.NLLLoss):
#     """ Online hard example mining.
#     Needs input from nn.LogSotmax() """
#
#     def __init__(self, ratio):
#         super(NLL_OHEM, self).__init__(None, True)
#         self.ratio = ratio
#
#     def forward(self, x, y, ratio=None):
#         if ratio is not None:
#             self.ratio = ratio
#         num_inst = x.size(0)
#         num_hns = int(self.ratio * num_inst)
#         x_ = x.clone()
#         inst_losses = th.autograd.Variable(th.zeros(num_inst)).cuda()
#         for idx, label in enumerate(y.data):
#             inst_losses[idx] = -x_.data[idx, label]
#             # loss_incs = -x_.sum(1)
#         _, idxs = inst_losses.topk(num_hns)
#         x_hn = x.index_select(0, idxs)
#         y_hn = y.index_select(0, idxs)
#         return th.nn.functional.nll_loss(x_hn, y_hn)




def compute_loss_weights(output, target):
    epsilon = 1e-5
    m = len(target)
    _, pred = output.topk(1, 1, True, True)
    counters = [0, 0]
    for p in pred:
        counters[p] += 1
    return torch.FloatTensor([m/(2 * c + epsilon) for c in counters])