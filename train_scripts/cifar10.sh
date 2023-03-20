#DualFLSD-53
python train.py --dataset cifar10 --model resnet50 --loss dual_focal_loss_sd --gamma 3 --num_bins 15 -e 350 --seed 0 --save-path exp/dualfocallossSD53_seed0

#AdaDualFocalLoss
python train.py --dataset cifar10 --model resnet50 --loss adadualfocal --gamma_max 20 --adafocal_start_epoch 0 --num_bins 15 -e 350 --save-path exp/adadualfocal_seed0

#DualFocalLoss
python train.py --dataset cifar10 --model resnet50 --loss dual_focal_loss --gamma 5 --num_bins 15 -e 350 --seed 0 --save-path exp/dualfocalloss_5_seed0

#FocalLoss
python train.py --dataset cifar10 --model resnet50 --loss focal_loss --gamma 3 --num_bins 15 -e 350 --seed 0 --save-path exp/focalloss_3_seed0

#FocalLossSD
python train.py --dataset cifar10 --model resnet50 --loss focal_loss_sd --gamma 3 --num_bins 15 -e 350 --seed 0 --save-path exp/focallossSD53_seed0

#kde_ce setting
python train.py --dataset cifar10 --model resnet110 --loss dual_focal_loss --gamma 5 --decay 1e-4  --num_bins 15 -e 200 --seed 0 --save-path exp/kde_ce_setting_dual_focal_loss_test

# 20Newsgroup
python3 train.py --loss dual_focal_loss --gamma 3  --gamma_max 20 --adafocal_start_epoch 0 --num-epochs 50 --num-bins 15 --save-path exp/nlp_dualfocalloss