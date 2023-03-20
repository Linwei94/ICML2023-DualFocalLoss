'''
Script for training models.
'''

from torch import optim
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn

import random
import json
import sys
import os
import argparse
import collections
import math

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100
import Data.tiny_imagenet as tiny_imagenet
import Data.imagenet as imagenet

# Import network models
from Net.resnet import resnet50, resnet110
from Net.resnet_tiny_imagenet import resnet50 as resnet50_ti
from Net.resnet_tiny_imagenet import resnet110 as resnet110_ti
from Net.resnet_tiny_imagenet import resnet152 as resnet152_ti
from Net.wide_resnet import wide_resnet_cifar
from Net.densenet import densenet121

# Import loss functions
from Losses.loss import cross_entropy, focal_loss, focal_loss_sd
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score

# Import train and validation utilities
from train_utils import train_single_epoch, test_single_epoch

# Import validation metrics
from Metrics.metrics import test_classification_net, get_ece_kde
from Metrics.metrics import expected_calibration_error, maximum_calibration_error, adaECE_error, ClasswiseECELoss
from Metrics.metrics import l2_error
from Metrics.plots import reliability_plot, bin_strength_plot

# ImageNet: We use SGD as our optimiser with momentum of 0.9 and weight decay 10−4, 
# and train the models for 90 epochs with a learning rate of 0.01 for the first 30 epochs, 
# 0.001 for the next 30 epochs and 0.0001 for the last 30 epochs. We use a training batch size of 128. 
# We divide the 50,000 validation images into validation and test set of 25,000 images each.

dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
    'imagenet': 1000
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet,
    'imagenet': imagenet
}


models = {
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'resnet110_ti': resnet110_ti,
    'resnet152_ti': resnet152_ti,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121
}


def loss_function_save_name(loss_function,
                            scheduled=False,
                            gamma=1.0,
                            gamma1=1.0,
                            gamma2=1.0,
                            gamma3=1.0,
                            lamda=1.0):
    res_dict = {
        'cross_entropy': 'cross_entropy',
        'focal_loss': 'focal_loss_gamma_' + str(gamma),
        'focal_loss_sd': 'focal_loss_sd_gamma_' + str(gamma),
        'mmce': 'mmce_lamda_' + str(lamda),
        'mmce_weighted': 'mmce_weighted_lamda_' + str(lamda),
        'brier_score': 'brier_score',
        'adafocal': 'adafocal_' + str(gamma),
        'adadualfocal': 'adadualfocal_' + str(gamma),
    }
    if (loss_function == 'focal_loss' and scheduled == True):
        res_str = 'focal_loss_scheduled_gamma_' + str(gamma1) + '_' + str(gamma2) + '_' + str(gamma3)
    else:
        res_str = loss_function
    return res_str


def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    train_batch_size = 128
    test_batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    loss = "cross_entropy"
    gamma = 1.0
    gamma2 = 1.0
    gamma3 = 1.0
    lamda = 1.0
    weight_decay = 5e-4
    log_interval = 50
    save_interval = 50
    save_loc = './'
    model_name = None
    saved_model_name = "resnet50_cross_entropy_350.model"
    load_loc = './'
    model = "resnet50"
    epoch = 350
    first_milestone = 150 #Milestone for change in lr
    second_milestone = 250 #Milestone for change in lr
    gamma_schedule_step1 = 100
    gamma_schedule_step2 = 250

    parser = argparse.ArgumentParser(
        description="Training for calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)

    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("--load", action="store_true", dest="load",
                        help="Load from pretrained model")
    parser.set_defaults(load=False)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("-e", type=int, default=epoch, dest="epoch",
                        help='Number of training epochs')
    parser.add_argument("--lr", type=float, default=learning_rate,
                        dest="learning_rate", help='Learning rate')
    parser.add_argument("--mom", type=float, default=momentum,
                        dest="momentum", help='Momentum')
    parser.add_argument("--nesterov", action="store_true", dest="nesterov",
                        help="Whether to use nesterov momentum in SGD")
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=weight_decay,
                        dest="weight_decay", help="Weight Decay")
    parser.add_argument("--opt", type=str, default=optimiser,
                        dest="optimiser",
                        help='Choice of optimisation algorithm')

    parser.add_argument("--loss", type=str, default=loss, dest="loss_function",
                        help="Loss function to be used for training")
    parser.add_argument("--loss-mean", action="store_true", dest="loss_mean",
                        help="whether to take mean of loss instead of sum to train")
    parser.set_defaults(loss_mean=False)
    parser.add_argument("--gamma", type=float, default=gamma,
                        dest="gamma", help="Gamma for focal components")
    parser.add_argument("--gamma2", type=float, default=gamma2,
                        dest="gamma2", help="Gamma for different focal components")
    parser.add_argument("--gamma3", type=float, default=gamma3,
                        dest="gamma3", help="Gamma for different focal components")
    parser.add_argument("--lamda", type=float, default=lamda,
                        dest="lamda", help="Regularization factor")
    parser.add_argument("--gamma-schedule", type=int, default=0,
                        dest="gamma_schedule", help="Schedule gamma or not")
    parser.add_argument("--gamma-schedule-step1", type=int, default=gamma_schedule_step1,
                        dest="gamma_schedule_step1", help="1st step for gamma schedule")
    parser.add_argument("--gamma-schedule-step2", type=int, default=gamma_schedule_step2,
                        dest="gamma_schedule_step2", help="2nd step for gamma schedule")

    parser.add_argument("--log-interval", type=int, default=log_interval,
                        dest="log_interval", help="Log Interval on Terminal")
    parser.add_argument("--save-interval", type=int, default=save_interval,
                        dest="save_interval", help="Save Interval on Terminal")
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to export the model')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name",
                        help='name of the model')
    parser.add_argument("--load-path", type=str, default=load_loc,
                        dest="load_loc",
                        help='Path to load the model from')

    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to train')
    parser.add_argument("--first-milestone", type=int, default=first_milestone,
                        dest="first_milestone", help="First milestone to change lr")
    parser.add_argument("--second-milestone", type=int, default=second_milestone,
                        dest="second_milestone", help="Second milestone to change lr")
    # AdaFocal
    parser.add_argument("--num_bins", nargs="+", type=int, default=[15], dest="num_bins", help="Number of calibration bins")
    parser.add_argument("--binning", type=str, default='adaptive', dest="binning", help='Type of binning: fixed or adaptive')
    parser.add_argument("--gamma_lambda", type=float, default=1.0, dest="gamma_lambda", help="lambda for auto adaptive focal gamma = gamma0*exp(lambda*ece)")
    parser.add_argument("--gamma_max", type=float, default=1e10, dest="gamma_max", help="Maximum cutoff value for clipping exploding gammas")
    parser.add_argument("--adafocal_start_epoch", type=int, default=0, dest="adafocal_start_epoch", help="Epoch to start the sample adaptive focal calibration")
    parser.add_argument("--seed", type=int, default=0, dest="seed", help="torch.manual_seed()")

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    torch.manual_seed(args.seed)
    if args.dataset == 'imagenet':
        # ImageNet: We use SGD as our optimiser with momentum of 0.9 and weight decay 10−4, 
        # and train the models for 90 epochs with a learning rate of 0.01 for the first 30 epochs, 
        # 0.001 for the next 30 epochs and 0.0001 for the last 30 epochs. We use a training batch size of 128. 
        # We divide the 50,000 validation images into validation and test set of 25,000 images each.
        args.learning_rate = 0.01
        args.momentum = 0.9
        args.weight_decay = 1e-4
        args.train_batch_size = 128
        args.first_milestone = 30
        args.second_milestone = 60


    cuda = False
    if (torch.cuda.is_available() and args.gpu):
        cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](num_classes=num_classes)

    # Setting model name
    if args.model_name is None:
        args.model_name = args.model


    if args.gpu is True:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    start_epoch = 0
    num_epochs = args.epoch
    if args.load:
        net.load_state_dict(torch.load(args.load_loc + '/' + args.saved_model_name))
        start_epoch = int(args.saved_model_name[args.saved_model_name.rfind('_')+1:args.saved_model_name.rfind('.model')])

    if args.optimiser == "sgd":
        opt_params = net.parameters()
        optimizer = optim.SGD(opt_params,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optimiser == "adam":
        opt_params = net.parameters()
        optimizer = optim.Adam(opt_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1)

    if (args.dataset == 'tiny_imagenet'):
        train_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='train',
            batch_size=args.train_batch_size,
            pin_memory=args.gpu)

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
    elif (args.dataset == 'imagenet'):
        # ImageNet: We use SGD as our optimiser with momentum of 0.9 and weight decay 10−4, 
        # and train the models for 90 epochs with a learning rate of 0.01 for the first 30 epochs, 
        # 0.001 for the next 30 epochs and 0.0001 for the last 30 epochs. We use a training batch size of 128. 
        # We divide the 50,000 validation images into validation and test set of 25,000 images each.
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu
        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu
        )
    else:
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu
        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu
        )

    training_set_loss = {}
    val_set_loss = {}
    test_set_loss = {}
    val_set_err = {}

    for epoch in range(0, start_epoch):
        scheduler.step()

    
    # Initialize the prev_epoch_adabin_dict for epoch=0 (if using adafocal)
    prev_epoch_adabin_dict = collections.defaultdict(dict)
    default_num_bins = args.num_bins[0]
    for bin_no in range(default_num_bins):
        bin_lower, bin_upper = bin_no*(1/default_num_bins), (bin_no+1)*(1/default_num_bins)
        prev_epoch_adabin_dict[bin_no]['lower_bound'] = bin_lower
        prev_epoch_adabin_dict[bin_no]['upper_bound'] = bin_upper
        prev_epoch_adabin_dict[bin_no]['prop_in_bin'] = 1/default_num_bins
        prev_epoch_adabin_dict[bin_no]['accuracy_in_bin'] = (bin_lower+bin_upper)/2.0
        prev_epoch_adabin_dict[bin_no]['avg_confidence_in_bin'] = (bin_lower+bin_upper)/2.0
        prev_epoch_adabin_dict[bin_no]['ece'] = prev_epoch_adabin_dict[bin_no]['avg_confidence_in_bin'] - prev_epoch_adabin_dict[bin_no]['accuracy_in_bin']
        prev_epoch_adabin_dict[bin_no]['gamma_next_epoch'] = args.gamma

    best_val_acc = 0
    for epoch in range(start_epoch, num_epochs):
        if (args.loss_function == 'focal_loss' and args.gamma_schedule == 1):
            if (epoch < args.gamma_schedule_step1):
                gamma = args.gamma
            elif (epoch >= args.gamma_schedule_step1 and epoch < args.gamma_schedule_step2):
                gamma = args.gamma2
            else:
                gamma = args.gamma3
        else:
            gamma = args.gamma

        train_loss = train_single_epoch(epoch,
                                        net,
                                        train_loader,
                                        optimizer,
                                        device,
                                        loss_function=args.loss_function,
                                        gamma=gamma,
                                        lamda=args.lamda,
                                        loss_mean=args.loss_mean,
                                        prev_epoch_adabin_dict=prev_epoch_adabin_dict,
                                        gamma_lambda=args.gamma_lambda,
                                        adafocal_start_epoch=args.adafocal_start_epoch)
        scheduler.step()
        

        # fake val_loss and test_loss for now
        val_loss, test_loss = 0, 0
        # evaluate on val set and test set
        for num_bins in args.num_bins:
            # Evaluate val set
            val_confusion_matrix, val_acc, val_labels, val_predictions, val_confidences, val_logits = \
                                            test_classification_net(net, val_loader, device, num_bins=num_bins, num_labels=num_classes)
            val_ece, val_bin_dict = expected_calibration_error(val_confidences, val_predictions, val_labels, num_bins=num_bins)
            val_mce = maximum_calibration_error(val_confidences, val_predictions, val_labels, num_bins=num_bins)
            val_adaece, val_adabin_dict = adaECE_error(val_confidences, val_predictions, val_labels, num_bins=num_bins)
            val_classwise_ece = ClasswiseECELoss(n_bins=num_bins)(val_logits, torch.tensor(val_labels))
            # Update the gamma for the next epoch
            if 'adafocal' in args.loss_function and epoch+1 >= args.adafocal_start_epoch:
                if args.binning == 'adaptive':
                    for bin_num in range(num_bins):
                        next_gamma = prev_epoch_adabin_dict[bin_num]['gamma_next_epoch'] * math.exp(args.gamma_lambda * val_adabin_dict[bin_num]['ece'])
                        val_adabin_dict[bin_num]['gamma_next_epoch'] = min(next_gamma, args.gamma_max)
                    prev_epoch_adabin_dict = val_adabin_dict
                elif args.binning == 'fixed':
                    for bin_num in range(num_bins):
                        next_gamma = prev_epoch_adabin_dict[bin_num]['gamma_next_epoch'] * math.exp(args.gamma_lambda * val_bin_dict[bin_num]['ece'])
                        val_bin_dict[bin_num]['gamma_next_epoch'] = min(next_gamma, args.gamma_max)
                    prev_epoch_adabin_dict = val_bin_dict

            # Evaluate test set
            test_confusion_matrix, test_acc, test_labels, test_predictions, test_confidences, test_logits = \
                                            test_classification_net(net, test_loader, device, num_bins=num_bins, num_labels=num_classes)
            test_ece, test_bin_dict = expected_calibration_error(test_confidences, test_predictions, test_labels, num_bins=num_bins)
            test_mce = maximum_calibration_error(test_confidences, test_predictions, test_labels, num_bins=num_bins)
            test_adaece, test_adabin_dict = adaECE_error(test_confidences, test_predictions, test_labels, num_bins=num_bins)
            test_classwise_ece = ClasswiseECELoss(n_bins=num_bins)(test_logits, torch.tensor(test_labels))
            test_ece_kde = get_ece_kde(F.softmax(test_logits, dim=1).cuda(), torch.tensor(test_labels).cuda(), bandwidth=0.001, p=1, mc_type='canonical', device='cuda')
            

            print('======> Test set ACC: {:.4f}'.format(test_acc))
            print('======> Test set ECE: {:.4f}'.format(test_ece))
            print('======> Test set MCE: {:.4f}'.format(test_mce))
            print('======> Test set adaECE: {:.4f}'.format(test_adaece))
            print('======> Test set classwise ECE: {:.4f}'.format(test_classwise_ece))
            print('======> Test set ECE KDE: {:.4f}'.format(test_ece_kde))


            # Metric logging
            training_set_loss[epoch] = train_loss
            val_set_loss[epoch] = val_loss
            test_set_loss[epoch] = test_loss
            val_set_err[epoch] = 1 - val_acc
            output_train_file = os.path.join(args.save_loc, "train_log_"+str(num_bins)+"bins.txt")
            if not os.path.isdir(args.save_loc):
                os.mkdir(args.save_loc)
            with open(output_train_file, "a") as writer:
                # epoch, train_loss, val_loss, test_loss, val_error, val_ece, val_mce, val_classwise_ece, test_error, test_ece, test_mce, test_classwise_ece
                writer.write("%d\t" % (epoch))
                writer.write("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (train_loss, val_loss, test_loss, 1 - val_acc, val_ece, val_mce, val_classwise_ece, val_adaece))
                writer.write("%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (1 - test_acc, test_ece, test_mce, test_classwise_ece, test_adaece))
                writer.write("\n")

            # Save the val_bin_dict, test_bin_dict to json files
            val_bin_dict_file = os.path.join(args.save_loc, "val_bin_dict_"+str(num_bins)+"bins.txt")
            with open(val_bin_dict_file, "a") as write_file:
                json.dump(val_bin_dict, write_file) 
                write_file.write("\n")
            test_bin_dict_file = os.path.join(args.save_loc, "test_bin_dict_"+str(num_bins)+"bins.txt")
            with open(test_bin_dict_file, "a") as write_file:
                json.dump(test_bin_dict, write_file) 
                write_file.write("\n")

            # Save the val_adabin_dict, test_adabin_dict to json files
            val_adabin_dict_file = os.path.join(args.save_loc, "val_adabin_dict_"+str(num_bins)+"bins.txt")
            with open(val_adabin_dict_file, "a") as write_file:
                json.dump(val_adabin_dict, write_file) 
                write_file.write("\n")
            test_adabin_dict_file = os.path.join(args.save_loc, "test_adabin_dict_"+str(num_bins)+"bins.txt")
            with open(test_adabin_dict_file, "a") as write_file:
                json.dump(test_adabin_dict, write_file) 
                write_file.write("\n")

    
    # save the final weight
    save_name = args.save_loc + '/' + \
                        args.model_name + '_' + \
                        loss_function_save_name(args.loss_function, args.gamma_schedule, gamma, args.gamma, args.gamma2, args.gamma3, args.lamda) + \
                        '_' + "final" + '.model'
    torch.save(net.state_dict(), save_name)