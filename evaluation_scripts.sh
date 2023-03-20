#!/bin/bash

# CIFAR10 resnet50
# -------------------------------------------------------------------------------------------
# Weight Decay
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_cross_entropy_350.model
# # MMCE
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_mmce_weighted_lamda_2.0_350.model
# # Brier Loss
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_brier_score_350.model
# # Label Smoothing
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_cross_entropy_smoothed_smoothing_0.05_350.model
# # FLSD-53
# python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet50_focal_loss_adaptive_53_350.model

# CIFAR10 resnet110
# # -------------------------------------------------------------------------------------------
# # Weight Decay
# python3 evaluate.py -log --dataset cifar10 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet110_cross_entropy_350.model
# # MMCE
# python3 evaluate.py -log --dataset cifar10 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet110_mmce_weighted_lamda_2.0_350.model
# # Brier Loss
# python3 evaluate.py -log --dataset cifar10 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet110_brier_score_350.model
# # Label Smoothing
# python3 evaluate.py -log --dataset cifar10 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet110_cross_entropy_smoothed_smoothing_0.05_350.model
# # FLSD-53
# python3 evaluate.py -log --dataset cifar10 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name resnet110_focal_loss_adaptive_53_350.model

# # CIFAR10 wideresnet
# -------------------------------------------------------------------------------------------
# # Weight Decay
# python3 evaluate.py -log --dataset cifar10 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name wide_resnet_cross_entropy_350.model
# # MMCE
# python3 evaluate.py -log --dataset cifar10 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name wide_resnet_mmce_weighted_lamda_2.0_350.model
# # Brier Loss
# python3 evaluate.py -log --dataset cifar10 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name wide_resnet_brier_score_350.model
# # Label Smoothing
# python3 evaluate.py -log --dataset cifar10 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name wide_resnet_cross_entropy_smoothed_smoothing_0.05_350.model
# # FLSD-53
# python3 evaluate.py -log --dataset cifar10 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name wide_resnet_focal_loss_adaptive_53_350.model

# # CIFAR10 densenet121
# # -------------------------------------------------------------------------------------------
# # Weight Decay
# python3 evaluate.py -log --dataset cifar10 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name densenet121_cross_entropy_350.model
# # MMCE
# python3 evaluate.py -log --dataset cifar10 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name densenet121_mmce_weighted_lamda_2.0_350.model
# # Brier Loss
# python3 evaluate.py -log --dataset cifar10 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name densenet121_brier_score_350.model
# # Label Smoothing
# python3 evaluate.py -log --dataset cifar10 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name densenet121_cross_entropy_smoothed_smoothing_0.05_350.model
# # FLSD-53
# python3 evaluate.py -log --dataset cifar10 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar10/ --saved_model_name densenet121_focal_loss_adaptive_53_350.model








# # CIFAR100 resnet50
# # -------------------------------------------------------------------------------------------
# # Weight Decay
# echo "resnet50 cross entropy"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet50_cross_entropy_350.model
# # MMCE
# echo "resnet50 mmce"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet50_mmce_weighted_lamda_2.0_350.model
# # Brier Loss
# echo "resnet50 brier score"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet50_brier_score_350.model
# # Label Smoothing
# echo "resnet50 label smoothing"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet50_cross_entropy_smoothed_smoothing_0.05_350.model
# # FLSD-53
# echo "resnet50 focal loss"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet50_focal_loss_adaptive_53_350.model

# # CIFAR100 resnet110
# # -------------------------------------------------------------------------------------------
# # Weight Decay
# echo "resnet110 Weight Decay"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet110_cross_entropy_350.model
# # MMCE
# echo "resnet110 MMCE"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet110_mmce_weighted_lamda_2.0_350.model
# # Brier Loss
# echo "resnet110 Brier Loss"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet110_brier_score_430.model
# # Label Smoothing
# echo "resnet110 Label Smoothing"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet110_cross_entropy_smoothed_smoothing_0.05_350.model
# # FLSD-53
# echo "resnet110 FLSD-53"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name resnet110_focal_loss_adaptive_53_350.model

# # CIFAR100 wideresnet
# # -------------------------------------------------------------------------------------------
# # Weight Decay
# echo "wideresnet Weight Decay"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name wide_resnet_cross_entropy_350.model
# # MMCE
# echo "wideresnet MMCE"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name wide_resnet_mmce_weighted_lamda_2.0_350.model
# # Brier Loss
# echo "wideresnet Brier Loss"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name wide_resnet_brier_score_350.model
# # Label Smoothing
# echo "wideresnet Label Smoothing"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name wide_resnet_cross_entropy_smoothed_smoothing_0.05_350.model
# # FLSD-53
# echo "wideresnet FLSD-53"
# echo "_________________________________________________________"
# python3 evaluate.py -log --dataset cifar100 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name wide_resnet_focal_loss_adaptive_53_350.model

# # CIFAR100 densenet121
# # -------------------------------------------------------------------------------------------
# # Weight Decay
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_cross_entropy_350.model
# # MMCE
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_mmce_weighted_lamda_2.0_350.model
# # Brier Loss
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_brier_score_350.model
# # Label Smoothing
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_cross_entropy_smoothed_smoothing_0.05_350.model
# # FLSD-53
# python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/cifar100/ --saved_model_name densenet121_focal_loss_adaptive_53_350.model




echo "cifar10 resnet50 dual focal"
echo "_________________________________________________________"
python3 evaluate.py -log --dataset cifar10 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/dualfocal/ --saved_model_name cifar10-resnet50.pt
echo "cifar10 resnet110 dual focal"
echo "_________________________________________________________"
python3 evaluate.py -log --dataset cifar10 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/dualfocal/ --saved_model_name cifar10-resnet110.pt
echo "cifar10 wideresnet dual focal"
echo "_________________________________________________________"
python3 evaluate.py -log --dataset cifar10 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/dualfocal/ --saved_model_name cifar10-wide_resnet.pt
echo "cifar10 densenet121 dual focal"
echo "_________________________________________________________"
python3 evaluate.py -log --dataset cifar10 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/dualfocal/ --saved_model_name cifar10-densenet121.pt
echo "cifar100 resnet50 dual focal"
echo "_________________________________________________________"
python3 evaluate.py -log --dataset cifar100 --model resnet50 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/dualfocal/ --saved_model_name cifar100-resnet50.pt
echo "cifar100 resnet110 dual focal"
echo "_________________________________________________________"
python3 evaluate.py -log --dataset cifar100 --model resnet110 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/dualfocal/ --saved_model_name cifar100-resnet110.pt
echo "cifar100 wideresnet dual focal"
echo "_________________________________________________________"
python3 evaluate.py -log --dataset cifar100 --model wide_resnet --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/dualfocal/ --saved_model_name cifar100-wide_resnet.pt
echo "cifar100 densenet121 dual focal"
echo "_________________________________________________________"
python3 evaluate.py -log --dataset cifar100 --model densenet121 --save-path /home/linwei/Desktop/projects/DualFocalLoss/weights/dualfocal/ --saved_model_name cifar100-densenet121.pt