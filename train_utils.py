'''
This module contains methods for training models with different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn

from Losses.loss import cross_entropy, focal_loss, focal_loss_sd
from Losses.loss import dual_focal_loss, dual_focal_loss_sd
from Losses.loss import dual_focal_loss_ablation1, dual_focal_loss_ablation2, dual_focal_loss_ablation3, dual_focal_loss_ablation4, dual_focal_loss_ablation5, dual_focal_loss_ablation6
from Losses.loss import adafocal, adadualfocal
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score
# from Losses.loss import kde_ce


loss_function_dict = {
    'cross_entropy': cross_entropy,
    'focal_loss': focal_loss,
    'focal_loss_sd': focal_loss_sd,
    'mmce': mmce,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score,
    'adafocal': adafocal,
    'adadualfocal': adadualfocal,
    'dual_focal_loss': dual_focal_loss,
    'dual_focal_loss_sd': dual_focal_loss_sd,
    'dual_focal_loss_ablation1': dual_focal_loss_ablation1,
    'dual_focal_loss_ablation2': dual_focal_loss_ablation2,
    'dual_focal_loss_ablation3': dual_focal_loss_ablation3,
    'dual_focal_loss_ablation4': dual_focal_loss_ablation4,
    'dual_focal_loss_ablation5': dual_focal_loss_ablation5,
    'dual_focal_loss_ablation6': dual_focal_loss_ablation6,

    # 'kde_ce': kde_ce
}


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       loss_function='cross_entropy',
                       gamma=1.0,
                       lamda=1.0,
                       loss_mean=False,
                       prev_epoch_adabin_dict=None,
                       gamma_lambda=1.0,
                       adafocal_start_epoch=0):
    '''
    Util method for training a model for a single epoch.
    '''
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # Shape of data = [Batch, Seq-len] = torch.Size([128, 1000])
        logits = model(data) # logits = torch.Size([128, 20]) = [Batch, Classes]

        if ('mmce' in loss_function):
            loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device))
        else:
            loss = loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device, 
                                                      prev_epoch_adabin_dict=prev_epoch_adabin_dict, gamma_lambda=gamma_lambda, adafocal_start_epoch=adafocal_start_epoch, epoch=epoch)

        if loss_mean:
            loss = loss / len(data)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average training loss: {:.4f}'.format(epoch, train_loss / num_samples))
    return train_loss / num_samples



def test_single_epoch(epoch,
                      model,
                      test_val_loader,
                      device,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0,
                      prev_epoch_adabin_dict=None,
                      gamma_lambda=1.0,
                      adafocal_start_epoch=0):
    '''
    Util method for testing a model for a single epoch.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            if ('mmce' in loss_function):
                loss += (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item())
            else:
                loss += loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device, 
                                                prev_epoch_adabin_dict=prev_epoch_adabin_dict, gamma_lambda=gamma_lambda, adafocal_start_epoch=adafocal_start_epoch, epoch=epoch).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(loss / num_samples))
    return loss / num_samples