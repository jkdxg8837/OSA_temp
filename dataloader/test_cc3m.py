
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision.models import resnet50
from torchvision import datasets, transforms
import webdataset as wds
import dataclasses
import time
from collections import deque
from typing import Optional





bucket = "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main"
# trainset_url = bucket + "/cc3m-train-{0000..0575}.tar"
trainset_url = bucket + "/cc3m-train-{0000..0005}.tar"
valset_url = bucket + "/cc3m-val-{0000..0002}.tar"
batch_size = 32
cache_dir="./cache"
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
def make_sample(sample, val=False):
    """Take a decoded sample dictionary, augment it, and return an (image, label) tuple."""
    assert not val, "only implemented training dataset for this notebook"
    image = sample["jpg"]
    label = sample["txt"]
    return transform_train(image), label

trainset = wds.WebDataset(trainset_url, resampled=True, shardshuffle=True, cache_dir=cache_dir, nodesplitter=wds.split_by_node)
# valset = wds.WebDataset(valset_url, resampled=True, shardshuffle=True, cache_dir=cache_dir, nodesplitter=wds.split_by_node)
trainset = trainset.shuffle(1000).decode("pil").map(make_sample)

# For IterableDataset objects, the batching needs to happen in the dataset.
trainset = trainset.batched(64)
trainloader = wds.WebLoader(trainset, batch_size=None, num_workers=4, shuffle=False)
trainloader = trainloader.with_epoch(1282 * 100 // 64)

os.environ["GOPEN_VERBOSE"] = "1"
images, texts = next(iter(trainloader))
print(texts[0])
os.environ["GOPEN_VERBOSE"] = "0"