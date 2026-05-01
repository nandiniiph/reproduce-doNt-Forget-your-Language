#!/bin/bash

BASE_PATH=$(pwd)

echo "===== JIGSAW IDEAL MODEL ====="
mkdir -p outputs/jigsaw/ideal

python src/nfl.py \
--train_file data/unbiased_jigsaw_balance_train.csv \
--model_name roberta-base \
--reg_method None \
--seed 24 \
--output_dir outputs/jigsaw/ideal \
| tee outputs/jigsaw/ideal/log.txt


echo "===== JIGSAW DFR 5% ====="
mkdir -p outputs/jigsaw/dfr

python src/dfr.py \
--train_file data/unbiased_jigsaw_balance_train.csv \
--data_percentage 0.05 \
--model_name $BASE_PATH/outputs/jigsaw/none/checkpoint-7944 \
--test_name jigsaw_balance_test \
--seed 24 \
| tee outputs/jigsaw/dfr/log.txt


echo "===== JIGSAW DFR 100% ====="

python src/dfr.py \
--train_file data/unbiased_jigsaw_balance_train.csv \
--data_percentage 1.0 \
--model_name $BASE_PATH/outputs/jigsaw/none/checkpoint-7944 \
--test_name jigsaw_balance_test \
--seed 24 \
| tee -a outputs/jigsaw/dfr/log.txt


echo "===== DONE JIGSAW IDEAL + DFR ====="