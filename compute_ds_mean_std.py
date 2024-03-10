from data import *

# Create ImageFolder dataset
dataloader = get_dataloader(
    f"/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/dataset/train",
    1,
    False,
    drop_last=True,
)

# Initialize variables to store cumulative sums
mean = torch.zeros([3])
std = torch.zeros([3])
num_images = 0

ct = 0
# Compute the mean and standard deviation
for batch, _ in dataloader:
    batch_mean = batch.mean(dim=(0, 2, 3))
    batch_std = batch.std(dim=(0, 2, 3))

    mean += batch_mean
    std += batch_std
    num_images += batch.size(0)
    ct += 1

# Compute overall mean and standard deviation
mean /= ct
std /= ct

print("Mean of dataset images (RGB):", mean)
print("Standard deviation of dataset images (RGB):", std)
