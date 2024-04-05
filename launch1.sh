#!/bin/bash
export CUDA_VISIBLE_DEVICE=1

TRAIN_DATA_PATH="/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/dataset/train"
MODEL_SAVE_PATH="/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/AI6102Project_ImageClassification/model_ckpt/resnet101_nopret"
TRANSFORM="default_transform"
# TRANSFORM="canny_transform"
EPOCHS=25
FREEZE_BACKBONE="n"

python main.py --model "resnet101" \
               --train_data_path "$TRAIN_DATA_PATH" \
               --model_save_path "$MODEL_SAVE_PATH" \
               --transform "$TRANSFORM" \
               --epochs "$EPOCHS" \
               --lr 0.0004 \
               --scheduler_type "cos_annealing" \
               --freeze_backbone "$FREEZE_BACKBONE" \
               # --pretrained_weight "resnet101"

# python main.py --model "ViT" \
#                --train_data_path "$TRAIN_DATA_PATH" \
#                --model_save_path "$MODEL_SAVE_PATH" \
#                --transform "$TRANSFORM" \
#                --epochs "$EPOCHS" \
#                --freeze_backbone "$FREEZE_BACKBONE"
