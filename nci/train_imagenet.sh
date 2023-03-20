#!/bin/bash

#PBS -P li96
#PBS -q gpuvolta

#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=32GB
#PBS -l jobfs=256GB
#PBS -l walltime=00:40:00
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l wd

# load modules
module load pytorch/1.10.0

# extract dataset
DATASET_ROOT="/scratch/li96/yl8645/data"
echo "DATASET_ROOT: ${DATASET_ROOT}"

# load dataset (ImageNet-1K)
PRJ_DATA_ROOT="${SCRATCH_ROOT}/data"
echo "DATASET_ROOT: ${DATASET_ROOT}"

mkdir -p ${PBS_JOBFS}/imagenet

echo "[$(date)] - Extract ImageNet-1K start" 
tar -xf ${DATASET_ROOT}/imagenet/imagenet2012.tar -C ${PBS_JOBFS}/imagenet
echo "[$(date)] - Extract ImageNet-1K done"

echo "[$(date)] - Build val folder start"
cp ${DATASET_ROOT}/imagenet/imagenet-build-val-folder.sh ${PBS_JOBFS}/imagenet/ImageNet2012/val
chmod u+x ${PBS_JOBFS}/imagenet/ImageNet2012/val/imagenet-build-val-folder.sh
cd ${PBS_JOBFS}/imagenet/ImageNet2012/val && bash ./imagenet-build-val-folder.sh
echo "[$(date)] - Build val folder done"



# train
DATASET="imagenet"
DATA_ROOT="${PBS_JOBFS}/imagenet/ImageNet2012"
MODEL="resnet50"
LOSS="dual_focal_loss"
GAMMA=5
SEED=0


cd /scratch/li96/lt2442/DualFocalLoss
python3 train.py --dataset ${DATASET} --dataset-root= ${DATA_ROOT} --model ${MODEL} --loss ${LOSS} --gamma ${GAMMA} --num_bins 15 -e 350 --seed ${SEED} --save-path exp/${DATASET}_${MODEL}_${LOSS}_gamma${GAMMA}_seed${SEED}

