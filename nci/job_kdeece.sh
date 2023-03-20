#!/bin/bash

#PBS -l ncpus=12
#PBS -l mem=30GB
#PBS -l jobfs=200GB
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -P li96
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l wd

module load python3/3.9.2
module load pytorch/1.9.0
cd /scratch/li96/lt2442/DualFocalLoss
python3 train.py --dataset cifar10 --model resnet110 --loss dual_focal_loss --gamma ${GAMMA} --decay 1e-4  --num_bins 15 -e 200 --seed 0 --save-path exp/kde_ce_setting_dual_focal_loss_${GAMMA}