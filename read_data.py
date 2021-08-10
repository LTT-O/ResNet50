"""
用于已下载数据集的转换,便于pytorch的读取
"""
import os

import torch
from torch.utils.data import DataLoader
import config
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomResizedCrop(448),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# nw = min([os.cpu_count(), config.BATCH_SIZE if config.BATCH_SIZE > 1 else 0, 8])

nw = 0
def train_data_load():
    # 训练集

    train_dataset = datasets.ImageFolder(root=config.ROOT_TRAIN, transform=train_transform)
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                                               num_workers=nw)

    return train_num, train_loader


def test_data_load():
    # 测试集
    validate_dataset = datasets.ImageFolder(root=config.ROOT_TEST, transform=test_transform)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                                  num_workers=nw)
    return val_num, validate_loader


