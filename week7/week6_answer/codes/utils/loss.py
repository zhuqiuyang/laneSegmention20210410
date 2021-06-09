import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MySoftmaxCrossEntropyLoss(nn.Module):

    def __init__(self, nbclasses):
        super(MySoftmaxCrossEntropyLoss, self).__init__()
        self.nbclasses = nbclasses

    def forward(self, inputs, target):
        #print(inputs.shape, target.shape)  #torch.Size([2, 8, 384, 1024]) torch.Size([2, 384, 1024])

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, self.nbclasses)  # N,H*W,C => N*H*W,C
        target = target.view(-1)
        return nn.CrossEntropyLoss(reduction="mean")(inputs, target)


class DiceLoss(nn.Module):

    def __init__(self, nbclasses):
        super(DiceLoss, self).__init__()
        self.nbclasses = nbclasses
        self.eps = 1e-6

    def forward(self, inputs, target):
        #print(inputs.shape, target.shape)  #torch.Size([2, 8, 384, 1024]) torch.Size([2, 384, 1024])
        target = target.unsqueeze(1)
        target0 = torch.zeros(target.shape[0], self.nbclasses, target.shape[2], target.shape[3]).cuda()
        target0 = target0.scatter_(1, target, 1) #one-hot

        inter = (inputs * target0).sum(-1).sum(-1)
        summation = (inputs + target0).sum(-1).sum(-1)
        dice_loss = 1 - 2 * inter / (summation + self.eps)
        return dice_loss.mean()
