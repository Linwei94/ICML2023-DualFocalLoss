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
python3 train.py --dataset ${DATASET} --model ${MODEL} --loss ${LOSS} --gamma ${GAMMA} --gamma_lambda ${LAMDA} --gamma_max 20 --adafocal_start_epoch 0 --num_bins 15 -e 350 --seed ${SEED} --save-path exp/${DATASET}_${MODEL}_${LOSS}_gamma${GAMMA}_lamda${LAMDA}_seed${SEED}