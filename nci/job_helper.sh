#!/bin/bash

LOSSES=("adadualfocal")
GAMMAS=(3 4 5 6 7 8 9)
LAMDAS=(0.1 0.2 0.5 1 2 3 4 5 10)
SEEDS=(0)
DATASETS=("cifar10" "cifar100")
MODELS=("resnet50" "resnet110" "densenet121" "wide_resnet")

for LOSS in "${LOSSES[@]}"
  do
    for GAMMA in "${GAMMAS[@]}"
      do
        for LAMDA in "${LAMDAS[@]}"
          do
            for SEED in "${SEEDS[@]}"
              do
                for DATASET in "${DATASETS[@]}"
                  do
                    for MODEL in "${MODELS[@]}"
                      do
                        qsub -N "${LOSS}-GAMMA${GAMMA}-LAMDA${LAMDA}-SEED${SEED}-DATASET${DATASET}-MODEL${MODEL}" -v LOSS="${LOSS}",SEED="${SEED}",GAMMA="${GAMMA}",LAMDA="${LAMDA}",DATASET="${DATASET}",MODEL="${MODEL}" job.sh
                      done
                  done
              done
          done
      done
  done



# GAMMAS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15)

# for GAMMA in "${GAMMAS[@]}"
#   do
#     qsub -N "KDEECE-GAMMA${GAMMA}" -v GAMMA="${GAMMA}" job_kdeece.sh
#   done