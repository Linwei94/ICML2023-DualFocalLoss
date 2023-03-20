from torch.nn import functional as F
from Losses.focal_loss import FocalLoss
from Losses.dual_focal_loss import DualFocalLoss
from Losses.dual_focal_loss_ablation import DualFocalLossAblation1, DualFocalLossAblation2, DualFocalLossAblation3, DualFocalLossAblation4, DualFocalLossAblation5, DualFocalLossAblation6
from Losses.focal_loss_sd import FocalLossSD
from Losses.dual_focal_loss_sd import DualFocalLossSD
from Losses.adafocal import AdaFocal
from Losses.adadualfocal import AdaDualFocal
from Losses.mmce import MMCE, MMCE_weighted
from Losses.brier_score import BrierScore
# from Metrics.metrics import get_bandwidth, get_ece_reg


def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, reduction='sum')

def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs['gamma'])(logits, targets)

def dual_focal_loss(logits, targets, **kwargs):
    return DualFocalLoss(gamma=kwargs['gamma'])(logits, targets)

def dual_focal_loss_ablation1(logits, targets, **kwargs):
    return DualFocalLossAblation1(gamma=kwargs['gamma'])(logits, targets)

def dual_focal_loss_ablation2(logits, targets, **kwargs):
    return DualFocalLossAblation2(gamma=kwargs['gamma'])(logits, targets)

def dual_focal_loss_ablation3(logits, targets, **kwargs):
    return DualFocalLossAblation3(gamma=kwargs['gamma'])(logits, targets)

def dual_focal_loss_ablation4(logits, targets, **kwargs):
    return DualFocalLossAblation4(gamma=kwargs['gamma'])(logits, targets)

def dual_focal_loss_ablation5(logits, targets, **kwargs):
    return DualFocalLossAblation5(gamma=kwargs['gamma'])(logits, targets)

def dual_focal_loss_ablation6(logits, targets, **kwargs):
    return DualFocalLossAblation6(gamma=kwargs['gamma'])(logits, targets)

def focal_loss_sd(logits, targets, **kwargs):
    return FocalLossSD(gamma=kwargs['gamma'], device=kwargs['device'])(logits, targets)

def dual_focal_loss_sd(logits, targets, **kwargs):
    return DualFocalLossSD(gamma=kwargs['gamma'], device=kwargs['device'])(logits, targets)

def adafocal(logits, targets, **kwargs):
    return AdaFocal(gamma=kwargs['gamma'], device=kwargs['device'], prev_epoch_adabin_dict=kwargs['prev_epoch_adabin_dict'], gamma_lambda=kwargs['gamma_lambda'],
                                    adafocal_start_epoch=kwargs['adafocal_start_epoch'], epoch=kwargs['epoch'])(logits, targets)

def adadualfocal(logits, targets, **kwargs):
    return AdaDualFocal(gamma=kwargs['gamma'], device=kwargs['device'], prev_epoch_adabin_dict=kwargs['prev_epoch_adabin_dict'], gamma_lambda=kwargs['gamma_lambda'],
                                    adafocal_start_epoch=kwargs['adafocal_start_epoch'], epoch=kwargs['epoch'])(logits, targets)

def mmce(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)

def mmce_weighted(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE_weighted(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)

def brier_score(logits, targets, **kwargs):
    return BrierScore()(logits, targets)

# def kde_ce(logits, targets, **kwargs):
#     ce = cross_entropy(logits, targets, **kwargs)
#     bandwidth = get_bandwidth(b='auto', f=F.softmax(logits, dim=1), device='cuda')
#     print(bandwidth)
#     reg = get_ece_reg(f=F.softmax(logits, dim=1), y=targets, bandwidth=bandwidth, p=1, mc_type='canonical', device='cuda')

#     return ce + kwargs['lamda'] * reg