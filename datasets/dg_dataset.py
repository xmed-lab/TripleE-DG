import cv2
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
import numpy as np
from skimage.transform import resize
from PIL import Image
import pandas as pd
from os.path import join, dirname
from random import sample, random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import bisect
import torch
import torchvision
from data_utils import fourier_augmentation

import os

# lib
def get_split_dataset_info(txt_list, val_percentage):


    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()
    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels

def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def isMulti(self):
        return isinstance(self.datasets[0], JigsawTestDatasetMultiple)

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx], dataset_idx

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class BaselineDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None, tile_transformer=None, two_transform=False):
        self.data_path = ""
        self.names = names
        self.labels = labels
        self.two_transform = two_transform

        # self.names = ['photo_dog.jpg',
        #               'art_elephant.jpg',
        #               'dog_style.jpg']
        self.N = len(self.names)


        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def get_image(self, index):
        framename = self.data_path + self.names[index]

        img = Image.open(framename).convert('RGB')
        if self._augment_tile is not None:
            if self.two_transform:
                img1 = self._augment_tile(self._image_transformer(img))
                img2 = self._augment_tile(self._image_transformer(img))
                img = torch.cat([img1, img2], 0)
                return img
            else:
                return self._augment_tile(self._image_transformer(img))
        else:
            return self._image_transformer(img), framename

    def __getitem__(self, index):
        img, framename = self.get_image(index)
        return img, int(self.labels[index]), framename

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('data/permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

class JigsawDataset(data.Dataset):
    def __init__(self, names, labels, jig_classes=30, img_transformer=None, tile_transformer=None, patches=True,
                 bias_whole_image=None, args=None, cropsize=224):
        self.data_path = ""
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image

        self.baug = args.baug
        self.ESAugS = args.ESAugS
        self.ESAugF = args.ESAugF

        self.ifstylize = False
        self.cropsize = cropsize
        self.ifstylize_box = []

        if patches:
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)

            self.returnFunc = make_grid

    def get_tile(self, img, n):

        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def get_image(self, index):

        framename = self.data_path + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = img.resize((self.cropsize, self.cropsize)) 
        return img
    
    def get_ref_image(self):

        index = np.random.random_integers(len(self.names) - 1) 
        framename = self.data_path + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = img.resize((self.cropsize, self.cropsize)) 
        return img

    def __getitem__(self, index):


        img_tr = [transforms.RandomResizedCrop((224, 224), (0.8, 1.0))]
        transform_train_onlycrop = transforms.Compose(img_tr)


        image = self.get_image(index)
        ref_image = self.get_ref_image()
        img_box = []
        index_dic = []


        if self.ESAugF:
            # ESAug-Fourier: add Fourier
            for i in range(0, 2*self.baug):
                index_dic.append(False)
                if np.random.random() <= float(1/15):  # modify to -1 to disable Fourier
                    aug_image, _ = fourier_augmentation(image, ref_image, 'AM', np.random.random(), np.random.random())
                    img_box.append(self._augment_tile(self._image_transformer(aug_image)))  # ESAugF1
                    # img_box.append(self._augment_tile((aug_image))) # ESAugF2
                    # img_box.append(self._augment_tile(transform_train_onlycrop(aug_image)))  # ESAugF3
                    # img_box.append(self._augment_tile(transform_train_onlycrop(aug_image)))  # ESAugF4: with two random ratios
                else:
                    img_box.append(self._augment_tile(self._image_transformer(image)))

        elif self.ESAugS:
            # ESAug-style: add Style
            for i in range(0, 2 * self.baug):
                if np.random.random() <= float(1/15):
                    index_dic.append(True)
                    img_box.append(self._augment_tile(transform_train_onlycrop(image)))
                else:
                    index_dic.append(False)
                    img_box.append(self._augment_tile(self._image_transformer(image)))
        else:
            # no ESAug: orignal augmentation
            for i in range(0, 2*self.baug):
                img_box.append(self._augment_tile(self._image_transformer(image)))
                index_dic.append(False)

        return img_box, self.labels[index], index_dic

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('data/permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm



if __name__ == '__main__':
    count = 0
    tot_count = 0
