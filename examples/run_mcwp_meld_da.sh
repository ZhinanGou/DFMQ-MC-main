#!/usr/bin bash

for dataset in 'MELD-DA'
do
    for seed in 0 
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'mcwp' \
        --method 'mcwp' \
        --data_mode 'multi-class' \
        --tune \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '0' \
        --video_feats_path 'swin_feats.pkl' \
        --audio_feats_path 'wavlm_feats.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'mcwp_meld_da' \
        --results_file_name 'results_mcwp_meld_da.csv'
    done
done