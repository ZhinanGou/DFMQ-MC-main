#!/usr/bin/bash

for seed in 0
do
    for method in 'mcwp' 
    do
        for text_backbone in 'bert-base-uncased'
        do
            python run.py \
            --dataset 'MIntRec' \
            --logger_name ${method} \
            --method ${method} \
            --tune \
            --train \
            --save_results \
            --seed $seed \
            --gpu_id '0' \
            --text_backbone $text_backbone \
            --video_feats_path "video_feats.pkl"\
            --audio_feats_path "audio_feats.pkl"\
            --config_file_name "mcwp_mintrec" \
            --results_file_name "results_mcwp_mintrec.csv"
        done
    done
done