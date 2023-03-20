# AdaFocal: Calibration-aware Adaptive Focal Loss

This repository is the official implementation of "AdaFocal: Calibration-aware Adaptive Focal Loss". 

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

## Training

To train on image classification tasks (CIFAR-10/100, Tiny-ImageNet) run:

```train
python train.py --dataset cifar10 --model resnet50 --loss cross_entropy --num_bins 15 -e 350 --save-path exp
python train.py --dataset cifar10 --model resnet50 --loss focal_loss --gamma 1 --num_bins 15 -e 350 --save-path exp
python train.py --dataset cifar10 --model resnet50 --loss focal_loss_sd --gamma 3 --num_bins 15 -e 350 --save-path exp
python train.py --dataset cifar10 --model resnet50 --loss adafocal --gamma_max 20 --adafocal_start_epoch 0 --num_bins 15 -e 350 --save-path exp
```

To train on 20 Newsgroup, run from inside the 20newsgroup directory:
```train
python train.py --loss cross_entropy --num-epochs 50 --num-bins 15 --save-path exp
python train.py --loss focal_loss --gamma 3 --num-epochs 50 --num-bins 15 --save-path exp
python train.py --loss focal_loss_sd --gamma 3 --num-epochs 50 --num-bins 15 --save-path exp
python train.py --loss adafocal --gamma_max 20 --adafocal_start_epoch 0 --num-epochs 50 --num-bins 15 --save-path exp
```

## Evaluation

To evaluate on image classification tasks (CIFAR-10/100, Tiny-ImageNet), run:

```eval
python evaluate.py -log --dataset cifar10 --model resnet50 --save-path /path/to/saved/model --saved_model_name model_name.model
```

To evaluate on on 20 Newsgroup, run from inside the 20newsgroup directory:

```eval
python evaluate.py -log --save-path /path/to/saved/model --saved_model_name model_name.model
```

## Pre-trained Models

Some of the pretrained models are provided in the "pretrained" directory.

## Results

AdaFocal achieves the following test set ECE(%) and Error(%) performance in comaparison to cross entropy, Brier Loss, MMCE, label smoothing, FLSD-53:

![ECE_Error_Result](ece_error_performance.PNG)



