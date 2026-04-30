#!/bin/bash

echo "===== BASELINE ====="
mkdir -p outputs/amazon/none
python src/nfl.py \
--train_file data/biased_amazon_train.csv \
--model_name roberta-base \
--reg_method None \
--seed 24 \
--output_dir outputs/amazon/none \
| tee outputs/amazon/none/log.txt


echo "===== NFL-F ====="
mkdir -p outputs/amazon/nfl_f
python src/nfl.py \
--train_file data/biased_amazon_train.csv \
--model_name roberta-base \
--reg_method NFL-F \
--reg_factor 0.1 \
--seed 24 \
--output_dir outputs/amazon/nfl_f \
| tee outputs/amazon/nfl_f/log.txt


echo "===== NFL-CO ====="
mkdir -p outputs/amazon/nfl_co
python src/nfl.py \
--train_file data/biased_amazon_train.csv \
--model_name roberta-base \
--reg_method NFL-CO \
--reg_factor 0.1 \
--seed 24 \
--output_dir outputs/amazon/nfl_co \
| tee outputs/amazon/nfl_co/log.txt


echo "===== NFL-CP ====="
mkdir -p outputs/amazon/nfl_cp
python src/nfl.py \
--train_file data/biased_amazon_train.csv \
--model_name roberta-base \
--reg_method NFL-CP \
--reg_factor 0.1 \
--seed 24 \
--output_dir outputs/amazon/nfl_cp \
| tee outputs/amazon/nfl_cp/log.txt


echo "===== NFL-PT ====="
mkdir -p outputs/amazon/nfl_pt
python src/nfl.py \
--train_file data/biased_amazon_train.csv \
--model_name roberta-base \
--reg_method NFL-PT \
--reg_factor 0.1 \
--seed 24 \
--output_dir outputs/amazon/nfl_pt \
| tee outputs/amazon/nfl_pt/log.txt


echo "===== DONE ALL ====="

