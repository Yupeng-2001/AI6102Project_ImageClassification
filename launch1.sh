#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# /mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/dataset/train
# /mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/unlabled_full_dataset
# /mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/dataset/train
TRAIN_DATA_PATH="/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/dataset/train"
MODEL_SAVE_PATH="/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/AI6102Project_ImageClassification/model_ckpt/swin_halfLR_v2"
# TRANSFORM="canny_transform"

# base LR = 0.0004
python main.py --model "swin" \
               --train_data_path "$TRAIN_DATA_PATH" \
               --model_save_path "$MODEL_SAVE_PATH" \
               --transform "default_pad" \
               --epochs 50 \
               --batch_size 64\
               --lr 0.0002 \
               --scheduler_type "cos_annealing" \
               --freeze_backbone "n" \
               --pretrained_weight "resnet101" \
               --dataset_mode default \
               --criterion ce \
               --seed 777\
               --use_feature

