from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

#define image transformations for preprocessing and augmentation
def get_transforms(train=True):
    base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]

    if train:
        base.insert(2, transforms.RandomHorizontalFlip())

    return transforms.Compose(base)

#define custom dataset class
class ChestXrayDataset(Dataset):
    def __init__(self, split_file, transform=None):
        self.samples = []
        self.transform = transform

        with open(split_file, "r") as f:
            for line in f:
                path, label = line.strip().split(",")
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label

