from typing import List

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
import torch

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: List[int]=[1]) -> List[torch.Tensor]:
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def binary_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions
    output and target are Torch tensors
    """
    pred = output.cpu() >= 0.5
    target = target.cpu()
    # print(list(output.data.cpu().numpy()))
    # print(list(pred.data[0].numpy()))
    # print(list(target.data[0].numpy()))
    # print(pred.size(), target.size())
    acc = (pred.int()).eq(target.int()).sum()
    acc = acc * 100 / np.prod(np.array(target.size()))
    return acc


def multiclass_metric(output, target):
    """
    Return balanced accuracy score (average of recall for each class) in case of class imbalance,
    and classification report containing precision, recall, F1 score for each class
    """
    balanced_acc = balanced_accuracy_score(target, output)
    report = classification_report(target, output)
    return balanced_acc, report
