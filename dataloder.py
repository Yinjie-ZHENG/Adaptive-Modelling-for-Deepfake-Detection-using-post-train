# -*- coding:UTF-8 -*-
"""
dataset and  data reading
"""

import os
import sys
from PIL import Image
# import glob
# import json
# import functools
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import random

# from osgeo import gdal
# import albumentations as albu
# from skimage.color import gray2rgb
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
#
#
# from utils.arg_utils import *
# from utils.data_utils import *
# from utils.algorithm_utils import *
from MLclf import MLclf

from autoaug.augmentations import Augmentation
from autoaug.archive import fa_reduced_cifar10, autoaug_paper_cifar10, fa_reduced_imagenet
import autoaug.aug_transforms as aug

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

from dataset_loder.scoliosis_dataloder import ScoliosisDataset
from autoaug.cutout import Cutout


# 自定义Dataset类，__getitem__(self,index)每次返回(img1, img2, 0/1)
class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)  # 37个类别中任选一个
        should_get_same_class = random.randint(0, 1)  # 保证同类样本约占一半
        if should_get_same_class:
            while True:
                # 直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # 直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0]).convert('RGB')
        img1 = Image.open(img1_tuple[0]).convert('RGB')




        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=transforms.ToTensor()):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


class miniFFDataset(Dataset):
    def __init__(self, csv_path, transform=transforms.ToTensor()):
        self.csv_path = csv_path
        self.transform = transform
        df = pd.read_csv(self.csv_path)
        dic = {'real': 0, 'fake': 1}
        file = Series.to_numpy(df["path"])
        file = [i for i in file]
        label_np = Series.to_numpy(df['label'])
        number = []
        for i in range(len(label_np)):
            number.append(dic[label_np[i]])
        number = np.array(number)
        self.images = file
        self.target = number

        self.imgs = list(zip(self.images, self.target))
        self.class_to_tgt_idx = {'real': 0, 'fake': 1}

    def __getitem__(self, idx):
        # assert len(self.images[idx]) == len(self.target[idx])
        img_path, tgt = self.images[idx], self.target[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

    def __len__(self):
        return len(self.images)




class deepfake_faces_set(Dataset):
    def __init__(self, csv_path, train_val_split=[0.8,0.2], train=True, transform=transforms.ToTensor()):

        self.Train = train
        self.csv_path = csv_path
        self.transform = transform
        self.train_val_split = train_val_split


        df = pd.read_csv(self.csv_path)
        dic = {'REAL': 0, 'FAKE': 1}

        file = Series.to_numpy(df["videoname"])
        file = [i.split(".mp4")[0] + ".jpg" for i in file]
        file = [os.path.join("/hdd7/yinjie/deepfake_faces/faces_224", i) for i in file]

        file_train = file[:int(train_val_split[0] * len(file))]
        file_val = file[int(train_val_split[0] * len(file)):]

        label_np = Series.to_numpy(df['label'])
        number = []
        for i in range(len(label_np)):
            number.append(dic[label_np[i]])
        number = np.array(number)
        number_train = number[:int(train_val_split[0] * len(number))]
        number_val = number[int(train_val_split[0] * len(number)):]

        if self.Train:
            self.images = file_train#[:1000]
            self.target = number_train#[:1000]

        else:#测试，取一点数据加速
            self.images = file_val#[:10]
            self.target = number_val#[:10]

        self.imgs = list(zip(self.images,self.target))
        self.class_to_tgt_idx = {'REAL': 0, 'FAKE': 1}

    def __getitem__(self, idx):
        # assert len(self.images[idx]) == len(self.target[idx])
        img_path, tgt = self.images[idx], self.target[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

    def __len__(self):
        return len(self.images)


def load_dataset(data_config):
    if data_config.dataset == 'cifar10':
        training_transform = training_transforms()
        if data_config.autoaug:
            print('auto Augmentation the data !')
            training_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        train_dataset = torchvision.datasets.CIFAR10(root=data_config.data_path,
                                                     train=True,
                                                     transform=training_transform,
                                                     download=True)
        val_dataset = torchvision.datasets.CIFAR10(root=data_config.data_path,
                                                   train=False,
                                                   transform=validation_transforms(),
                                                   download=True)
        return train_dataset, val_dataset
    elif data_config.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_config.data_path,
                                                      train=True,
                                                      transform=training_transforms(),
                                                      download=True)
        val_dataset = torchvision.datasets.CIFAR100(root=data_config.data_path,
                                                    train=False,
                                                    transform=validation_transforms(),
                                                    download=True)
        return train_dataset, val_dataset

    elif data_config.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root=data_config.data_path,
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        val_dataset = torchvision.datasets.MNIST(root=data_config.data_path,
                                                 train=False,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
        return train_dataset, val_dataset


    elif data_config.dataset == 'tiny_imagenet':
        data_path = '/home/yinjie/FYP_/torch/dataset/tiny-imagenet-200'  # '/hdd7/yinjie/tiny-imagenet-200'   #'/hdd7/yinjie/tiny-imagenet-200-dropusse'

        train_dataset = TinyImageNet(data_path, train=True, transform=training_transforms())
        val_dataset = TinyImageNet(data_path, train=False, transform=validation_transforms())
        return train_dataset, val_dataset

    elif data_config.dataset == 'deepfake_faces':
        data_path = '/hdd7/yinjie/deepfake_faces'
        csv_path = data_path + "/metadata.csv"

        train_dataset = deepfake_faces_set(csv_path, train_val_split=data_config.train_val_split, train=True, transform=data_config.training_transforms)
        val_dataset = deepfake_faces_set(csv_path, train_val_split=data_config.train_val_split, train=False, transform=data_config.validation_transforms)

        return train_dataset, val_dataset

    elif data_config.dataset == 'deepfake_faces_siamese':
        data_path = '/hdd7/yinjie/deepfake_faces'
        csv_path = data_path + "/metadata.csv"
        #First develop dataset for loading
        train_dataset = deepfake_faces_set(csv_path, train_val_split=data_config.train_val_split, train=True, transform=data_config.training_transforms)
        val_dataset = deepfake_faces_set(csv_path, train_val_split=data_config.train_val_split, train=False, transform=data_config.validation_transforms)
        #Then utilize dataset to develop siamese dataset
        train_dataset_siamese = SiameseNetworkDataset(train_dataset, transform=data_config.training_transforms)
        val_dataset_siamese   = SiameseNetworkDataset(val_dataset, transform=data_config.validation_transforms)


        return train_dataset_siamese, val_dataset_siamese

    elif data_config.dataset == 'miniFFDataset':
        data_path = '/hdd7/yinjie/miniFF'
        train_path = data_path + "/train_metadata.csv"
        val_path = data_path + "/val_metadata.csv"
        #First develop dataset for loading
        train_dataset = miniFFDataset(train_path, transform=data_config.training_transforms)
        val_dataset = miniFFDataset(val_path, transform=data_config.validation_transforms)
        #Then utilize dataset to develop siamese dataset
        #train_dataset_siamese = SiameseNetworkDataset(train_dataset, transform=data_config.training_transforms)
        #val_dataset_siamese   = SiameseNetworkDataset(val_dataset, transform=data_config.validation_transforms)

        return train_dataset, val_dataset

    elif data_config.dataset == 'miniFFDataset_siamese':
        data_path = '/hdd7/yinjie/miniFF'
        train_path = data_path + "/train_metadata.csv"
        val_path = data_path + "/val_metadata.csv"
        #First develop dataset for loading
        train_dataset = miniFFDataset(train_path, transform=data_config.training_transforms)
        val_dataset = miniFFDataset(val_path, transform=data_config.validation_transforms)
        #Then utilize dataset to develop siamese dataset
        train_dataset_siamese = SiameseNetworkDataset(train_dataset, transform=data_config.training_transforms)
        val_dataset_siamese   = SiameseNetworkDataset(val_dataset, transform=data_config.validation_transforms)

        return train_dataset_siamese, val_dataset_siamese

    elif data_config.dataset == 'imagenet':
        traindir = data_config.data_path + '/imagenet/train'
        valdir = data_config.data_path + '/imagenet/val'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        jittering = aug.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4)
        lighting = aug.Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]])
        train_dataset = torchvision.datasets.ImageFolder(traindir,
                                                         transforms.Compose([
                                                             transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             jittering, lighting, normalize, ]))
        val_dataset = torchvision.datasets.ImageFolder(valdir,
                                                       transforms.Compose([
                                                           transforms.Resize(256),
                                                           transforms.RandomResizedCrop(224),
                                                           transforms.ToTensor(),
                                                           normalize, ]))
        return train_dataset, val_dataset


    else:
        raise Exception('unknown dataset: {}'.format(data_config.dataset))
