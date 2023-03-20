"""
# ImageNet: We use SGD as our optimiser with momentum of 0.9 and weight decay 10−4, 
# and train the models for 90 epochs with a learning rate of 0.01 for the first 30 epochs, 
# 0.001 for the next 30 epochs and 0.0001 for the last 30 epochs. We use a training batch size of 128. 
# We divide the 50,000 validation images into validation and test set of 25,000 images each.
"""
import os
import torch
import numpy as numpy

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

import os
import glob
from torch.utils.data import Dataset
from PIL import Image

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'



def get_data_loader(root,
                    batch_size,
                    split='train',
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the Tiny Imagenet dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - root: The root directory for TinyImagenet dataset
    - batch_size: how many samples per batch to load.
    - split: Can be train/val/test. For train we apply the data augmentation techniques.
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """


    # load the dataset
    data_dir = root

    if split == 'train':
        dataset = TinyImageNet(data_dir,
                               split=split,
                               transform=train_transform,
                               in_memory=True)
    elif split == 'train_val':
        dataset = TinyImageNet(data_dir,
                               split=split,
                               transform=val_test_transform,
                               in_memory=True)
    else:
        dataset = TinyImageNet(data_dir,
                               split='val',
                               transform=val_test_transform,
                               in_memory=True)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    return data_loader