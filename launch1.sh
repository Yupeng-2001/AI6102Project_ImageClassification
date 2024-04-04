export CUDA_VISIBLE_DEVICE=0

#!/bin/bash
TRAIN_DATA_PATH="/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/train_canny"
MODEL_SAVE_PATH="/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/AI6102Project_ImageClassification/model_ckpt/resnet50_canny"
# TRANSFORM="default_transform"
TRANSFORM="canny_transform"
EPOCHS=25
FREEZE_BACKBONE="y"

python main.py --model "resnet50" \
               --train_data_path "$TRAIN_DATA_PATH" \
               --model_save_path "$MODEL_SAVE_PATH" \
               --transform "$TRANSFORM" \
               --epochs "$EPOCHS" \
               --lr 0.0004 \
               --scheduler_type "cos_annealing" \
               # --freeze_backbone "$FREEZE_BACKBONE"

# python main.py --model "ViT" \
#                --train_data_path "$TRAIN_DATA_PATH" \
#                --model_save_path "$MODEL_SAVE_PATH" \
#                --transform "$TRANSFORM" \
#                --epochs "$EPOCHS" \
#                --freeze_backbone "$FREEZE_BACKBONE"
