#!/bin/bash
set -e

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BASE_PATH=$(pwd)
CKPT="$BASE_PATH/outputs/amazon/none/checkpoint-7002"

echo "===== IDEAL MODEL ====="
mkdir -p outputs/amazon/ideal

python src/nfl.py \
--train_file data/unbiased_amazon_train.csv \
--model_name roberta-base \
--reg_method None \
--seed 24 \
--output_dir outputs/amazon/ideal \
| tee outputs/amazon/ideal/log.txt

echo "===== DFR 5% ====="
mkdir -p outputs/amazon/dfr

python src/dfr.py \
--train_file data/unbiased_amazon_train.csv \
--data_percentage 0.05 \
--model_name $CKPT \
--test_name amazon_test \
--seed 24 \
| tee outputs/amazon/dfr/log.txt


echo "===== DFR 100% ====="

python src/dfr.py \
--train_file data/unbiased_amazon_train.csv \
--data_percentage 1.0 \
--model_name $CKPT \
--test_name amazon_test \
--seed 24 \
| tee -a outputs/amazon/dfr/log.txt


echo "===== DONE IDEAL + DFR ====="