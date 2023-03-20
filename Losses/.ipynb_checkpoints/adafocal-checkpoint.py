'''
Implementation of Auto Adaptive Focal Loss.
Arindam Ghosh
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

class AutoAdaptiveFocalLossV2(nn.Module):
    def __init__(self, gamma=1, size_average=False, device=None, prev_epoch_adabin_dict=None, gamma_lambda=1.0, adafocal_start_epoch=0, epoch=None):
        super(AutoAdaptiveFocalLossV2, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.gamma_lambda = gamma_lambda
        self.device = device
        self.prev_epoch_adabin_dict = prev_epoch_adabin_dict
        self.adafocal_start_epoch = adafocal_start_epoch
        self.epoch = epoch

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]

        bin_list = [] # [(lower, upper, stats_dict), ..., ]
        for bin_no, bin_stats in self.prev_epoch_adabin_dict.items():
            bin_list.append((bin_stats['lower_bound'], bin_stats['upper_bound'], bin_stats))
        
        try:
            bin_list.sort()
        except:
            print(bin_list)
            exit(0)

        # Select the focal-gamma for each sample based on which bin it falls into and the value of ece in there
        for i in range(batch_size):
            if self.epoch >= self.adafocal_start_epoch:
                pt_sample = pt[i].item()
                for bin_no, element in enumerate(bin_list):    # bin_list = [(lower, upper, stats_dict), ..., ]
                    bin_stats = element[2]
                    if bin_no==0 and pt_sample < bin_stats['upper_bound']:
                        break
                    elif bin_no==len(bin_list)-1 and pt_sample >= bin_stats['lower_bound']:
                        break
                    elif pt_sample >= bin_stats['lower_bound'] and pt_sample < bin_stats['upper_bound']:
                        break
                gamma_list.append(bin_stats['gamma_next_epoch'])
            else:
                gamma_list.append(self.gamma)

        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        gamma = self.get_gamma_list(pt)
        loss = -1 * (1-pt + 1e-20)**gamma * logpt # 1e-20 added for numerical stability 

        if self.size_average: return loss.mean()
        else: return loss.sum()
