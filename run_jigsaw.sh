#!/bin/bash

echo "===== JIGSAW BASELINE ====="
mkdir -p outputs/jigsaw/none
python src/nfl.py \
--train_file data/biased_jigsaw_balance_train.csv \
--model_name roberta-base \
--reg_method None \
--seed 24 \
--output_dir outputs/jigsaw/none \
| tee outputs/jigsaw/none/log.txt


echo "===== JIGSAW NFL-F ====="
mkdir -p outputs/jigsaw/nfl_f
python src/nfl.py \
--train_file data/biased_jigsaw_balance_train.csv \
--model_name roberta-base \
--reg_method NFL-F \
--reg_factor 0.1 \
--seed 24 \
--output_dir outputs/jigsaw/nfl_f \
| tee outputs/jigsaw/nfl_f/log.txt


echo "===== JIGSAW NFL-CO ====="
mkdir -p outputs/jigsaw/nfl_co
python src/nfl.py \
--train_file data/biased_jigsaw_balance_train.csv \
--model_name roberta-base \
--reg_method NFL-CO \
--reg_factor 0.1 \
--seed 24 \
--output_dir outputs/jigsaw/nfl_co \
| tee outputs/jigsaw/nfl_co/log.txt


echo "===== JIGSAW NFL-CP ====="
mkdir -p outputs/jigsaw/nfl_cp
python src/nfl.py \
--train_file data/biased_jigsaw_balance_train.csv \
--model_name roberta-base \
--reg_method NFL-CP \
--reg_factor 0.1 \
--seed 24 \
--output_dir outputs/jigsaw/nfl_cp \
| tee outputs/jigsaw/nfl_cp/log.txt


echo "===== JIGSAW NFL-PT ====="
mkdir -p outputs/jigsaw/nfl_pt
python src/nfl.py \
--train_file data/biased_jigsaw_balance_train.csv \
--model_name roberta-base \
--reg_method NFL-PT \
--reg_factor 0.1 \
--seed 24 \
--output_dir outputs/jigsaw/nfl_pt \
| tee outputs/jigsaw/nfl_pt/log.txt


echo "===== DONE JIGSAW BASELINE + NFL ====="