#!/bin/bash

#PBS -P li96
#PBS -q gpuvolta

#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l jobfs=256GB
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l wd

PREFIX="ID: ${PBS_JOBID} (evaluation)"

SCRATCH_ROOT="/scratch/li96/yl8645"
echo "SCRATCH_ROOT: ${SCRATCH_ROOT}"

echo "[$(date)] - ${PREFIX} - Job start" | tee -a ${SCRATCH_ROOT}/out.txt

# load modules
module purge
module load python3/3.9.2
module load pytorch/1.10.0
module unload pytorch
source ${SCRATCH_ROOT}/venv/timm-pt110/bin/activate
echo "VENV: ${SCRATCH_ROOT}/venv/timm-pt110/bin/activate"

# load dataset (ImageNet-1K)
PRJ_DATA_ROOT="${SCRATCH_ROOT}/data"
echo "PRJ_DATA_ROOT: ${PRJ_DATA_ROOT}"

mkdir -p ${PBS_JOBFS}/imagenet

echo "[$(date)] - ${PREFIX} - Extract ImageNet-1K start" | tee -a ${SCRATCH_ROOT}/out.txt
tar -xf ${PRJ_DATA_ROOT}/imagenet/imagenet2012.tar -C ${PBS_JOBFS}/imagenet
echo "[$(date)] - ${PREFIX} - Extract ImageNet-1K done" | tee -a ${SCRATCH_ROOT}/out.txt

echo "[$(date)] - ${PREFIX} - Build val folder start" | tee -a ${SCRATCH_ROOT}/out.txt
cp ${PRJ_DATA_ROOT}/imagenet/imagenet-build-val-folder.sh ${PBS_JOBFS}/imagenet/ImageNet2012/val
chmod u+x ${PBS_JOBFS}/imagenet/ImageNet2012/val/imagenet-build-val-folder.sh
cd ${PBS_JOBFS}/imagenet/ImageNet2012/val && bash ./imagenet-build-val-folder.sh
echo "[$(date)] - ${PREFIX} - Build val folder done" | tee -a ${SCRATCH_ROOT}/out.txt

# set train params
PRJ_OUT_ROOT="${SCRATCH_ROOT}/outputs/transformers-adapter/val"
echo "PRJ_OUT_ROOT: ${PRJ_OUT_ROOT}"

BS=100

# start run experiment
echo "[$(date)] - ${PREFIX} - Run code" | tee -a ${SCRATCH_ROOT}/out.txt

MODEL_NAME="vit_adapter_tiny_patch16_224"
CKPT_ROOT="${SCRATCH_ROOT}/outputs/transformers-adapter"
CKPT_PATH="${CKPT_ROOT}/20230128-023432-vit_adapter_tiny_patch16_224/output"
CKPT_NAMES=(
    "checkpoint-6.pth.tar"
    "checkpoint-7.pth.tar"
    "checkpoint-8.pth.tar"
    "checkpoint-9.pth.tar"
    "checkpoint-10.pth.tar"
)
for _NAME in "${CKPT_NAMES[@]}"; do

    ls -lh "${CKPT_PATH}/${_NAME}"

    echo "[$(date)] - ${PREFIX} - ${CKPT_PATH}/${_NAME}" | tee -a ${SCRATCH_ROOT}/out.txt

    cd ${SCRATCH_ROOT}/codes/transformers-adapter && python3 validate.py \
        ${PBS_JOBFS}/imagenet/ImageNet2012 \
        --split val \
        --model ${MODEL_NAME} \
        --checkpoint ${CKPT_PATH}/${_NAME} \
        --output ${PRJ_OUT_ROOT} \
        --batch_size 512

    cd ${SCRATCH_ROOT}/codes/transformers-adapter && python3 validate.py \
        ${PBS_JOBFS}/imagenet/ImageNet2012 \
        --split val \
        --no_prefetcher \
        --model ${MODEL_NAME} \
        --checkpoint ${CKPT_PATH}/${_NAME} \
        --output ${PRJ_OUT_ROOT} \
        --attack fgsm \
        --batch_size 90
done

echo "[$(date)] - ${PREFIX} - All done" | tee -a ${SCRATCH_ROOT}/out.txt
