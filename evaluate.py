import os
import sys
import torch
import numpy as np
import random
import argparse
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100
import Data.tiny_imagenet as tiny_imagenet

# Import network architectures
from Net.resnet_tiny_imagenet import resnet50 as resnet50_ti
from Net.resnet import resnet50, resnet110
from Net.wide_resnet import wide_resnet_cifar
from Net.densenet import densenet121

# Import metrics to compute
from Metrics.metrics import test_classification_net_logits, get_ece_kde, BS
from Metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss


# Import temperature scaling and NLL utilities
from temperature_scaling import ModelWithTemperature

# From Kumar et. al.
import calibration as cal

# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet
}

# Mapping model name to model function
models = {
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121
}


def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    model = 'resnet50'
    save_loc = './'
    saved_model_name = 'resnet50_cross_entropy_350.model'
    num_bins = 15
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")

    return parser.parse_args()


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels


if __name__ == "__main__":

    # Checking if GPU is available
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    # Setting additional parameters
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    args = parseArgs()

    if args.model_name is None:
        args.model_name = args.model

    dataset = args.dataset
    dataset_root = args.dataset_root
    model_name = args.model_name
    save_loc = args.save_loc
    saved_model_name = args.saved_model_name
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error

    # Taking input for the dataset
    num_classes = dataset_num_classes[dataset]
    if (args.dataset == 'tiny_imagenet'):
        val_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='train_val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)

        test_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)
    else:
        _, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu
        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu
        )

    model = models[model_name]

    net = model(num_classes=num_classes, temp=1.0)
    net.cuda()
    # try:
    #     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #     cudnn.benchmark = True
    #     net.load_state_dict(torch.load(args.save_loc + '/' + args.saved_model_name))
    # except:
    cudnn.benchmark = True
    net.load_state_dict(torch.load(args.save_loc + '/' + args.saved_model_name)["model"])

    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    logits, labels = get_logits_labels(test_loader, net)
    conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()
    p_ece_kde = get_ece_kde(F.softmax(logits, dim=1).cpu(), labels.cpu(), bandwidth=0.001, p=1, mc_type='canonical', device='cpu')
    p_rbs = BS(logits, labels)






# -------------------------------------------------------------------------------------------
# post temperature scaling
# -------------------------------------------------------------------------------------------

    # Printing the required evaluation metrics
    if args.log:
        # print (conf_matrix)
        print ('ECE: {:.2f}'.format(p_ece*100))
        print ('AdaECE: {:.2f}'.format(p_adaece*100))
        print ('Classwise ECE: {:.2f}'.format(p_cece*100))
        print ('Test error: {:.2f}'.format((1 - p_accuracy)*100))
        print ('RBS: {:.4f}'.format(p_rbs))
        print ('ECE KDE: {:.2f}'.format(p_ece_kde*100), '\n')
        # print ('Test NLL: {:.2f}'.format(p_nll))

    scaled_model = ModelWithTemperature(net, args.log)
    scaled_model.set_temperature(val_loader, cross_validate=cross_validation_error)
    T_opt = scaled_model.get_temperature()
    logits, labels = get_logits_labels(test_loader, scaled_model)
    conf_matrix, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()
    ece_kde = get_ece_kde(F.softmax(logits, dim=1).cpu(), labels.cpu(), bandwidth=0.001, p=1, mc_type='canonical', device='cpu')
    rbs = BS(logits, labels)

    if args.log:
        print ('\nOptimal temperature: {:.2f}'.format(T_opt))
        # print (conf_matrix)
        print ('ECE: {:.2f}'.format(ece*100))
        print ('AdaECE: {:.2f}'.format(adaece*100))
        print ('Classwise ECE: {:.2f}'.format(cece*100))
        print ('RBS: {:.4f}'.format(rbs))
        print ('ECE KDE: {:.2f}'.format(ece_kde*100))
        # print ('Test error: {:.2f}'.format((1 - accuracy)*100))
        # print ('Test NLL: {:.2f}'.format(nll))




# -------------------------------------------------------------------------------------------
# Weight Decay
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_cross_entropy_350.model
# ECE: 4.34
# AdaECE: 4.33
# Classwise ECE: 0.90
# Test error: 4.95
# RBS: 0.0922
# ECE KDE: 6.86 

# Before temperature - NLL: 0.351, ECE: 0.040
# Optimal temperature: 2.500
# After temperature - NLL: 0.179, ECE: 0.014

# Optimal temperature: 2.50
# ECE: 1.36
# AdaECE: 2.14
# Classwise ECE: 0.45
# RBS: 0.0824
# ECE KDE: 10.07
# -------------------------------------------------------------------------------------------
# MMCE
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_mmce_weighted_lamda_2.0_350.model
# ECE: 4.57
# AdaECE: 4.55
# Classwise ECE: 0.94
# Test error: 4.99
# RBS: 0.0938
# ECE KDE: 7.09 

# Before temperature - NLL: 0.429, ECE: 0.044
# Optimal temperature: 2.600
# After temperature - NLL: 0.210, ECE: 0.013

# Optimal temperature: 2.60
# ECE: 1.19
# AdaECE: 2.16
# Classwise ECE: 0.52
# RBS: 0.0835
# -------------------------------------------------------------------------------------------
# Brier Loss
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_brier_score_350.model
# ECE: 1.83
# AdaECE: 1.74
# Classwise ECE: 0.46
# Test error: 5.00
# RBS: 0.0815
# ECE KDE: 10.57 

# Before temperature - NLL: 0.179, ECE: 0.019
# Optimal temperature: 1.100
# After temperature - NLL: 0.177, ECE: 0.015

# Optimal temperature: 1.10
# ECE: 1.08
# AdaECE: 1.23
# Classwise ECE: 0.41
# RBS: 0.0806
# ECE KDE: 11.96
# -------------------------------------------------------------------------------------------
# Label Smoothing
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_cross_entropy_smoothed_smoothing_0.05_350.model
# ECE: 2.97
# AdaECE: 3.88
# Classwise ECE: 0.71
# Test error: 5.29
# RBS: 0.0943
# ECE KDE: 11.13 

# Before temperature - NLL: 0.259, ECE: 0.034
# Optimal temperature: 0.900
# After temperature - NLL: 0.257, ECE: 0.013

# Optimal temperature: 0.90
# ECE: 1.67
# AdaECE: 2.92
# Classwise ECE: 0.51
# RBS: 0.0954
# -------------------------------------------------------------------------------------------
# FLSD-53
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_focal_loss_adaptive_53_350.model
# ECE: 1.56
# AdaECE: 1.57
# Classwise ECE: 0.42
# Test error: 4.99
# RBS: 0.0801
# ECE KDE: 10.49 

# Before temperature - NLL: 0.162, ECE: 0.014
# Optimal temperature: 1.100
# After temperature - NLL: 0.162, ECE: 0.010

# Optimal temperature: 1.10
# ECE: 0.93
# AdaECE: 1.27
# Classwise ECE: 0.42
# RBS: 0.0792
# ECE KDE: 12.18
# -------------------------------------------------------------------------------------------
# dual_focal_loss_ablation4
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_dual_focal_loss_ablation4_350.model
# ECE: 0.54
# AdaECE: 0.38
# Classwise ECE: 0.38
# Test error: 5.01
# RBS: 0.28
# ECE KDE: 13.47 

# Before temperature - NLL: 0.156, ECE: 0.008
# Optimal temperature: 1.000
# After temperature - NLL: 0.156, ECE: 0.008

# Optimal temperature: 1.00
# ECE: 0.54
# AdaECE: 0.38
# Classwise ECE: 0.38
# RBS: 0.28
# ECE KDE: 13.47
# -------------------------------------------------------------------------------------------
# kde setting
# python3 evaluate.py -log --dataset cifar10 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet110_dual_focal_loss_kde15.model


# -------------------------------------------------------------------------------------------
# other setting
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10_resnet50_dual_focal_loss_ablation6_gamma5_lamda1_seed0 --saved_model_name resnet50_dual_focal_loss_ablation6_final.model




# CIFAR100
# -------------------------------------------------------------------------------------------
# Cross Entropy
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_cross_entropy_350.model
# ECE: 21.00
# AdaECE: 21.00
# Classwise ECE: 0.45
# Test error: 24.54
# RBS: 0.4457
# ECE KDE: 32.40 

# Before temperature - NLL: 2.018, ECE: 0.202
# Optimal temperature: 2.300
# After temperature - NLL: 1.171, ECE: 0.039

# Optimal temperature: 2.30
# ECE: 4.29
# AdaECE: 5.09
# Classwise ECE: 0.24
# RBS: 0.3684
# ECE KDE: 65.45
# -------------------------------------------------------------------------------------------
# Brier Loss
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_brier_score_350.model
# ECE: 5.24
# AdaECE: 5.04
# Classwise ECE: 0.20
# Test error: 23.73
# RBS: 0.3390
# ECE KDE: 53.54 

# Before temperature - NLL: 1.012, ECE: 0.055
# Optimal temperature: 1.100
# After temperature - NLL: 1.015, ECE: 0.028

# Optimal temperature: 1.10
# ECE: 2.27
# AdaECE: 2.58
# Classwise ECE: 0.21
# RBS: 0.3354
# ECE KDE: 60.48
# -------------------------------------------------------------------------------------------
# MMCE
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_mmce_weighted_lamda_2.0_350.model
# ECE: 19.11
# AdaECE: 19.11
# Classwise ECE: 0.42
# Test error: 23.98
# RBS: 0.4181
# ECE KDE: 31.25 

# Before temperature - NLL: 1.733, ECE: 0.198
# Optimal temperature: 2.100
# After temperature - NLL: 1.168, ECE: 0.031

# Optimal temperature: 2.10
# ECE: 3.14
# AdaECE: 3.10
# Classwise ECE: 0.24
# RBS: 0.3547
# ECE KDE: 65.78
# -------------------------------------------------------------------------------------------
# Label Smoothing
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_cross_entropy_smoothed_smoothing_0.05_350.model
# ECE: 12.88
# AdaECE: 12.83
# Classwise ECE: 0.29
# Test error: 24.05
# RBS: 0.4003
# ECE KDE: 86.37 

# Before temperature - NLL: 1.417, ECE: 0.130
# Optimal temperature: 1.200
# After temperature - NLL: 1.357, ECE: 0.071

# Optimal temperature: 1.20
# ECE: 7.37
# AdaECE: 8.91
# Classwise ECE: 0.23
# RBS: 0.3806
# ECE KDE: 74.32
# -------------------------------------------------------------------------------------------
# FLSD-53
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_focal_loss_adaptive_53_350.model
# ECE: 3.71
# AdaECE: 3.55
# Classwise ECE: 0.19
# Test error: 22.67
# RBS: 0.3218
# ECE KDE: 51.76 

# Before temperature - NLL: 0.875, ECE: 0.044
# Optimal temperature: 1.100
# After temperature - NLL: 0.877, ECE: 0.020

# Optimal temperature: 1.10
# ECE: 1.38
# AdaECE: 1.25
# Classwise ECE: 0.20
# RBS: 0.3202
# ECE KDE: 58.02
# -------------------------------------------------------------------------------------------


