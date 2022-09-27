#! /bin/bash

# predictor without augmentation
#python prediction/train_predictor.py --task=mortality 

# Retrieval-augmented predictor
for strategy in avg wavg svote wvote
do
    for num_doc in 5
    do
        python prediction/train_predictor.py --task=MP_IN --num_doc_for_augment 5 --augment_strategy $strategy --predictor_exp_name strategy"$strategy".doc"$num_doc"
    done
done