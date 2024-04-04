from torchvision import transforms

NUM_CLASSES = 121

# mean and variance for aspect ratio preserved resize only dataset
mean_tr = [0.9414, 0.9414, 0.9414]
std_tr = [0.1496, 0.1496, 0.1496]

default_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.3, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_tr, std=std_tr),
    ]
)

inference_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_tr, std=std_tr),
    ]
)

canny_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.3, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
    ]
)
