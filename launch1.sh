
#!/bin/bash
TRAIN_DATA_PATH="/content/train"
MODEL_SAVE_PATH="/content"
TRANSFORM="default_transform"
EPOCHS=50
FREEZE_BACKBONE="y"

python main.py --model "resnet50" \
               --train_data_path "$TRAIN_DATA_PATH" \
               --model_save_path "$MODEL_SAVE_PATH" \
               --transform "$TRANSFORM" \
               --epochs "$EPOCHS" \
               --freeze_backbone "$FREEZE_BACKBONE"

python main.py --model "ViT" \
               --train_data_path "$TRAIN_DATA_PATH" \
               --model_save_path "$MODEL_SAVE_PATH" \
               --transform "$TRANSFORM" \
               --epochs "$EPOCHS" \
               --freeze_backbone "$FREEZE_BACKBONE"
