#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        output1 = F.normalize(output1, dim=1)  # (bs, dim)  --->  (bs, dim)
        output2 = F.normalize(output2, dim=1)
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

if __name__ == '__main__':
    pass
    #embed1 = torch.FloatTensor(16, 2048)
    #embed2 = torch.FloatTensor(16, 2048)
    #label = torch.from_numpy(np.array([0.], dtype=np.float32))
    #criterion = ContrastiveLoss()
    #loss = criterion(embed1, embed2, label)

