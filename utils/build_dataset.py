import os
import os.path
import torch
import json
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode 
from torch.utils.data import Dataset
from timm.data import create_transform
from PIL import Image
import numpy as np
import random
from torch.utils.data import Subset
from collections import defaultdict
import torchvision.transforms.functional as TF
from PIL import Image
from utils.data_handler import ImageFolderWithIndex, FileListDataset, Flowers

class DataAugmentation:
    def __init__(self, weak_transform, strong_transform):
        self.transforms = [weak_transform, strong_transform]

    def __call__(self, x):
        images_weak = self.transforms[0](x)
        images_strong = self.transforms[1](x)
        return images_weak, images_strong

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    train_config_path = os.path.join("./json_files", 'dataset_catalog.json')
    with open(train_config_path, 'r') as train_config_file:
        catalog = json.load(train_config_file)
    assert args.dataset in catalog.keys(), "Dataset %s is not implemented"%args.data
    entry = catalog[args.dataset]
    if entry['type']=='imagefolder':
        if args.dataset == "flowers":
            dataset = Flowers(root=entry['path'], transform=transform, istrain=is_train)
        else:
            dataset = ImageFolderWithIndex(os.path.join(entry['path'], entry['train'] if is_train else entry['test']), 
                                        transform=transform,)              
    else:     
        path = entry['train'] if is_train else entry['test']
        image_file = os.path.join(entry['path'], path + '_images.npy')
        label_file = os.path.join(entry['path'], path + '_labels.npy')
        target_transform = None
        dataset = FileListDataset(image_file, label_file, transform, target_transform) 
    len_original = len(dataset)
    if args.dataset == "imagenet":
        if is_train:
            dataset = generate_few_shot(dataset, 50) 
            return dataset , len_original    
    return dataset, len_original

def generate_few_shot(dataset, num_shot):
    if num_shot < 0: 
        return dataset
    class_indices = defaultdict(list)
    for i, element in enumerate(dataset.samples):
        class_indices[element[1]].append(i)
    selected_indices = []
    for indices in class_indices.values():
        # Check if k is greater than the number of elements in the class
        if  num_shot> len(indices):
            print(f"Warning: Class has fewer than {num_shot} elements.")
            selected_indices.extend(indices)
        else:
            selected_indices.extend(random.sample(indices, num_shot))
    subset_dataset = Subset(dataset, selected_indices)    
    return subset_dataset

def build_transform(is_train, args):
    if is_train:
        weak_transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),       
            transforms.RandomCrop(args.input_size),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std))
        ])
        strong_transform = create_transform(
            input_size=args.input_size,
            scale=(args.train_crop_min,1),
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            mean=args.image_mean,
            std=args.image_std
        )         
        transform = DataAugmentation(weak_transform, strong_transform)
        return transform
    else:
        transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std))
        ])
        return transform