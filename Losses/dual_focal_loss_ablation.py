'''
Implementation of Focal Loss.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 1 - p_k + p_j + p_3_largest
class DualFocalLossAblation1(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(DualFocalLossAblation1, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        p_3_largest = torch.topk(p_j_mask * softmax_logits, 2)[0][:, 1].squeeze()
        p_4_largest = torch.topk(p_j_mask * softmax_logits, 3)[0][:, 2].squeeze()
        p_5_largest = torch.topk(p_j_mask * softmax_logits, 4)[0][:, 3].squeeze()
        p_6_largest = torch.topk(p_j_mask * softmax_logits, 5)[0][:, 4].squeeze()
        p_7_largest = torch.topk(p_j_mask * softmax_logits, 6)[0][:, 5].squeeze()
        p_8_largest = torch.topk(p_j_mask * softmax_logits, 7)[0][:, 6].squeeze()
        p_9_largest = torch.topk(p_j_mask * softmax_logits, 8)[0][:, 7].squeeze()
        p_10_largest = torch.topk(p_j_mask * softmax_logits, 9)[0][:, 8].squeeze()


        loss = -1 * (1 - p_k + p_j + p_3_largest) ** self.gamma * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()

# 1 - p_k + p_j + p_3_largest + p_4_largest
class DualFocalLossAblation2(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(DualFocalLossAblation2, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        p_3_largest = torch.topk(p_j_mask * softmax_logits, 2)[0][:, 1].squeeze()
        p_4_largest = torch.topk(p_j_mask * softmax_logits, 3)[0][:, 2].squeeze()
        p_5_largest = torch.topk(p_j_mask * softmax_logits, 4)[0][:, 3].squeeze()
        p_6_largest = torch.topk(p_j_mask * softmax_logits, 5)[0][:, 4].squeeze()
        p_7_largest = torch.topk(p_j_mask * softmax_logits, 6)[0][:, 5].squeeze()
        p_8_largest = torch.topk(p_j_mask * softmax_logits, 7)[0][:, 6].squeeze()
        p_9_largest = torch.topk(p_j_mask * softmax_logits, 8)[0][:, 7].squeeze()
        p_10_largest = torch.topk(p_j_mask * softmax_logits, 9)[0][:, 8].squeeze()


        loss = -1 * (1 - p_k + p_j + p_3_largest + p_4_largest) ** self.gamma * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()


# 1 - p_k + p_j + p_3_largest + p_4_largest + p_5_largest
class DualFocalLossAblation3(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(DualFocalLossAblation3, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        p_3_largest = torch.topk(p_j_mask * softmax_logits, 2)[0][:, 1].squeeze()
        p_4_largest = torch.topk(p_j_mask * softmax_logits, 3)[0][:, 2].squeeze()
        p_5_largest = torch.topk(p_j_mask * softmax_logits, 4)[0][:, 3].squeeze()
        p_6_largest = torch.topk(p_j_mask * softmax_logits, 5)[0][:, 4].squeeze()
        p_7_largest = torch.topk(p_j_mask * softmax_logits, 6)[0][:, 5].squeeze()
        p_8_largest = torch.topk(p_j_mask * softmax_logits, 7)[0][:, 6].squeeze()
        p_9_largest = torch.topk(p_j_mask * softmax_logits, 8)[0][:, 7].squeeze()
        p_10_largest = torch.topk(p_j_mask * softmax_logits, 9)[0][:, 8].squeeze()


        loss = -1 * (1 - p_k + p_j + p_3_largest + p_4_largest + p_5_largest) ** self.gamma * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()



# 1 - p_k + torch.mean(p_j + p_3_largest + p_4_largest + p_5_largest + p_6_largest + p_7_largest + p_8_largest + p_9_largest + p_10_largest)
class DualFocalLossAblation4(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(DualFocalLossAblation4, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        p_3_largest = torch.topk(p_j_mask * softmax_logits, 2)[0][:, 1].squeeze()
        p_4_largest = torch.topk(p_j_mask * softmax_logits, 3)[0][:, 2].squeeze()
        p_5_largest = torch.topk(p_j_mask * softmax_logits, 4)[0][:, 3].squeeze()
        p_6_largest = torch.topk(p_j_mask * softmax_logits, 5)[0][:, 4].squeeze()
        p_7_largest = torch.topk(p_j_mask * softmax_logits, 6)[0][:, 5].squeeze()
        p_8_largest = torch.topk(p_j_mask * softmax_logits, 7)[0][:, 6].squeeze()
        p_9_largest = torch.topk(p_j_mask * softmax_logits, 8)[0][:, 7].squeeze()
        p_10_largest = torch.topk(p_j_mask * softmax_logits, 9)[0][:, 8].squeeze()


        loss = -1 * (1 - p_k + torch.mean(p_j + p_3_largest + p_4_largest + p_5_largest + p_6_largest + p_7_largest + p_8_largest + p_9_largest + p_10_largest)) ** self.gamma * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()



# 1 - p_k + torch.mean(p_j + p_3_largest)
class DualFocalLossAblation5(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(DualFocalLossAblation5, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        p_3_largest = torch.topk(p_j_mask * softmax_logits, 2)[0][:, 1].squeeze()
        p_4_largest = torch.topk(p_j_mask * softmax_logits, 3)[0][:, 2].squeeze()
        p_5_largest = torch.topk(p_j_mask * softmax_logits, 4)[0][:, 3].squeeze()
        p_6_largest = torch.topk(p_j_mask * softmax_logits, 5)[0][:, 4].squeeze()
        p_7_largest = torch.topk(p_j_mask * softmax_logits, 6)[0][:, 5].squeeze()
        p_8_largest = torch.topk(p_j_mask * softmax_logits, 7)[0][:, 6].squeeze()
        p_9_largest = torch.topk(p_j_mask * softmax_logits, 8)[0][:, 7].squeeze()
        p_10_largest = torch.topk(p_j_mask * softmax_logits, 9)[0][:, 8].squeeze()


        loss = -1 * (1 - p_k + torch.mean(p_j + p_3_largest)) ** self.gamma * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()




# 1 - p_k + torch.mean(p_j + p_3_largest + p_4_largest)
class DualFocalLossAblation6(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(DualFocalLossAblation6, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        p_3_largest = torch.topk(p_j_mask * softmax_logits, 2)[0][:, 1].squeeze()
        p_4_largest = torch.topk(p_j_mask * softmax_logits, 3)[0][:, 2].squeeze()
        p_5_largest = torch.topk(p_j_mask * softmax_logits, 4)[0][:, 3].squeeze()
        p_6_largest = torch.topk(p_j_mask * softmax_logits, 5)[0][:, 4].squeeze()
        p_7_largest = torch.topk(p_j_mask * softmax_logits, 6)[0][:, 5].squeeze()
        p_8_largest = torch.topk(p_j_mask * softmax_logits, 7)[0][:, 6].squeeze()
        p_9_largest = torch.topk(p_j_mask * softmax_logits, 8)[0][:, 7].squeeze()
        p_10_largest = torch.topk(p_j_mask * softmax_logits, 9)[0][:, 8].squeeze()


        loss = -1 * (1 - p_k + torch.mean(p_j + p_3_largest + p_4_largest)) ** self.gamma * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()




        