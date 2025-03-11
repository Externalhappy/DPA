from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
import json

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class FileListDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
    def __getitem__(self, index):
        image = pil_loader(self.images[index])
        target = self.labels[index]
        if self.transform is not None:
            sample = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index
    def __len__(self):
        return len(self.images)

class ImageFolderWithIndex(datasets.ImageFolder):
    """Custom dataset that includes image file index. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithIndex, self).__getitem__(index)
        # make a new tuple that includes original and the path
        tuple_with_index = (original_tuple + (index,))
        return tuple_with_index

class Flowers(Dataset):
    def __init__(self, root, transform=None, istrain=True):
        image_dir = os.path.join(root, "jpg")
        split_path = os.path.join(root, "split_zhou_OxfordFlowers.json")
        self.data = self.read_split(split_path, image_dir, istrain)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, idx
    def read_split(self, filepath, path_prefix, istrain):
        def _convert(items):
            out = []
            for impath, label, _ in items:
                impath = os.path.join(path_prefix, impath)
                item = (impath, int(label))
                out.append(item)
            return out
        def read_json(fpath):
            """Read json file from a path."""
            with open(fpath, "r") as f:
                obj = json.load(f)
            return obj
        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        if istrain:
            return _convert(split["train"])
        return _convert(split["test"])