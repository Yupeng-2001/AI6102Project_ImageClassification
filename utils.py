import random
import numpy as np
import torch
import datetime
import os
import json


def reset_seeds(seed=42):
    """
    Reset seeds for reproducibility.

    Args:
    - seed: Integer seed value (default is 42).
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure reproducibility when using CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(
    model_save_path, model_type, trianing_result, best_val_loss, model, classes
):
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M")

    # generate model weight's file name and save
    model_filename = (
        "_".join([model_type, time_string, "valLoss:" + str(best_val_loss)[:8]])
        + ".pth"
    )

    model_file_path = os.path.join(model_save_path, model_filename)
    torch.save(model, model_file_path)

    # save the result of the train process
    # as well as classes that keeps track of class index mapping
    json_content = {}
    json_content["classes"] = classes
    json_content["trianing_process"] = trianing_result

    json_filename = "_".join([model_type, time_string]) + ".json"
    json_file_path = os.path.join(model_save_path, json_filename)
    with open(json_file_path, "w") as json_file:
        json.dump(json_content, json_file)
