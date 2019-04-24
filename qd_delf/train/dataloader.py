
#-*- coding: utf-8 -*-

'''
dataloader.py
'''

import sys, os, time

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageFile
from dataset import CropClassTSVDatasetYaml, CropClassTSVDatasetYamlList

Image.MAX_IMAGE_PIXELS = 1000000000     # to avoid error "https://github.com/zimeon/iiif/issues/11"
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid error "https://github.com/python-pillow/Pillow/issues/1510"

def get_loader(
    data_cfg_path,
    stage,
    train_batch_size,
    val_batch_size,
    sample_size,
    crop_size,
    workers):
    bgr_normalize = transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                     std=[0.225, 0.224, 0.229])

    if stage in ['finetune']:
        assert False

    elif stage in ['keypoint']:
        # for train
        train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(crop_size, scale=(0.25,1), ratio=(2./3., 3./2.)),
                    # transforms.ColorJitter(brightness=(0.66667, 1.5), contrast=0, saturation=(0.66667, 1.5), hue=(-0.1, 0.1)),
                    transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0.5, hue=0.1),
                    # transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    bgr_normalize,
                ])

        # for val
        test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(sample_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    bgr_normalize,
                ])
        enlarge_bbox = 2.0

    # image folder dataset.
    if data_cfg_path.endswith('.yaml'):
        train_dataset = CropClassTSVDatasetYaml(data_cfg_path, session_name='train', transform=train_transform, enlarge_bbox=enlarge_bbox)
    elif data_cfg_path.endswith('.yamllst'):
        train_dataset = CropClassTSVDatasetYamlList(data_cfg_path, session_name='train', transform=train_transform, enlarge_bbox=enlarge_bbox)
    else:
        raise NotImplementedError()
    if data_cfg_path.endswith('.yaml'):
        val_dataset = CropClassTSVDatasetYaml(data_cfg_path, session_name='val', transform=test_transform, enlarge_bbox=enlarge_bbox)
    elif data_cfg_path.endswith('.yamllst'):
        val_dataset = CropClassTSVDatasetYamlList(data_cfg_path, session_name='val', transform=test_transform, enlarge_bbox=enlarge_bbox)
    else:
        raise NotImplementedError()

    # return train/val dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = train_batch_size, shuffle = True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                             batch_size = val_batch_size,
                                             shuffle = False,
                                             num_workers = workers, pin_memory=True)

    return train_dataset, val_dataset, train_loader, val_loader

def get_loader1(
    train_path,
    val_path,
    stage,
    train_batch_size,
    val_batch_size,
    sample_size,
    crop_size,
    workers):

    if stage in ['finetune']:
        # for train
        prepro = []
        prepro.append(transforms.Resize(size=sample_size))
        prepro.append(transforms.CenterCrop(size=sample_size))
        prepro.append(transforms.RandomCrop(size=crop_size, padding=0))
        prepro.append(transforms.RandomHorizontalFlip())
        #prepro.append(transforms.RandomRotation((-15, 15)))        # experimental.
        prepro.append(transforms.ToTensor())
        train_transform = transforms.Compose(prepro)
        train_path = train_path

        # for val
        prepro = []
        prepro.append(transforms.Resize(size=sample_size))
        prepro.append(transforms.CenterCrop(size=crop_size))
        prepro.append(transforms.ToTensor())
        val_transform = transforms.Compose(prepro)
        val_path = val_path

    elif stage in ['keypoint']:
        # for train
        prepro = []
        prepro.append(transforms.Resize(size=sample_size))
        prepro.append(transforms.CenterCrop(size=sample_size))
        prepro.append(transforms.RandomCrop(size=crop_size, padding=0))
        prepro.append(transforms.RandomHorizontalFlip())
        #prepro.append(transforms.RandomRotation((-15, 15)))        # experimental.
        prepro.append(transforms.ToTensor())
        train_transform = transforms.Compose(prepro)
        train_path = train_path

        # for val
        prepro = []
        prepro.append(transforms.Resize(size=sample_size))
        prepro.append(transforms.CenterCrop(size=crop_size))
        prepro.append(transforms.ToTensor())
        val_transform = transforms.Compose(prepro)
        val_path = val_path

    # image folder dataset.
    train_dataset = datasets.ImageFolder(root = train_path,
                                         transform = train_transform)
    val_dataset = datasets.ImageFolder(root = val_path,
                                       transform = val_transform)

    # return train/val dataloader
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = train_batch_size,
                                               shuffle = True,
                                               num_workers = workers)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                             batch_size = val_batch_size,
                                             shuffle = False,
                                             num_workers = workers)

    return train_loader, val_loader



