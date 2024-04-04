import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from train import *
from data import *
from constants import *
import math


def inference(model, dataloader: DataLoader, outfile, device):
    model.eval()
    all_probabilities = None
    for imgs, _ in tqdm(dataloader):
        imgs = imgs.to(device)
        predictions = model(imgs)  # each: [bsz, num_cls]
        if all_probabilities is None:
            all_probabilities = predictions
        else:
            all_probabilities = torch.cat((all_probabilities, predictions), dim=0)
    return all_probabilities


transform = default_transform
BATCH_SIZE = 64

dl_tt = get_dataloader(
    f"/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/test_dataset",
    BATCH_SIZE,
    shuffle=False,
    transform=default_transform,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_path = "/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/AI6102Project_ImageClassification/model_ckpt/resnet50_2024-03-31_23-15_valLoss:0.862225.pth"
model_weights = torch.load(model_path, map_location=torch.device("cpu"))
# model = ResNetClassifier(num_classes=NUM_CLASSES)
# model.load_state_dict(model_weights)
model = model_weights
for param in model.parameters():
    param.requires_grad = False
model = model.to(device)
model.eval()
ret_pred = inference(model, dl_tt, None, device)
pass
