#!/bin/bash
SEED=0

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir /data/natural-instructions \
	    --val_data_dir /data/natural-instructions \
        --selection_seed ${SEED} \
        --max_steps 600 \
        --batch_size 4 \
        --slice_list stance_detection text_matching \
        --sample_rule mixture \
        --context_length 2048 \
        --proportions 1 0 \
        --num_ckpts 6 \
        --filter_val_skills \
        --debug_val \
        --icl \
        --session_id stance_detection_icl_test \
done 



python3 main.py \
    --task_name ni \
    --train_data_dir /share/kuleshov/jy928/skill-icl/data/natural-instructions \
    --val_data_dir /share/kuleshov/jy928/skill-icl/data/natural-instructions \
    --selection_seed 0 \
    --max_steps 600 \
    --batch_size 4 \
    --slice_list stance_detection text_matching \
    --sample_rule mixture \
    --context_length 2048 \
    --proportions 1 1 \
    --num_ckpts 6 \
    --filter_val_skills \
    --icl \
    --session_id stance_detection_icl_test_full_eval \