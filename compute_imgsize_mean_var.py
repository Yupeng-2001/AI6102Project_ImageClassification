from PIL import Image
import os
import numpy as np

import pdb

def get_image_dimensions(folder_path):
    widths = []
    heights = []
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                # Open the image file
                with Image.open(file_path) as img:
                    # Get dimensions
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
            except:
                # Handle exceptions if unable to open or process image
                print(f"Skipping file {file_path}")

    return widths, heights

def compute_mean_and_variance(folder_list):
    all_widths = []
    all_heights = []
    
    # Iterate over each folder
    for folder_path in folder_list:
        if os.path.isdir(folder_path):
            # Get dimensions of images in the folder
            widths, heights = get_image_dimensions(folder_path)
            all_widths.extend(widths)
            all_heights.extend(heights)

    # Compute mean and variance
    mean_width = np.mean(all_widths)
    mean_height = np.mean(all_heights)
    
    var_width = np.var(all_widths)
    var_height = np.var(all_heights)

    return mean_width, mean_height, var_width, var_height

# List of folders containing images
folder_list = ['/content/test', '/content/train']

# Compute mean and variance of image dimensions
mean_width, mean_height, var_width, var_height = compute_mean_and_variance(folder_list)

print(f"Mean Width: {mean_width}")
print(f"Mean Height: {mean_height}")
print(f"Variance of Width: {var_width}")
print(f"Variance of Height: {var_height}")