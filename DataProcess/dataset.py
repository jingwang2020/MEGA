# -*- coding: utf-8 -*-
# @Time    : 2021/8/12 16:39
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : dataset.py
# @Software: PyCharm
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import numpy as np


class MuliTsBatchedWindowDataset(Dataset):

    """
        Building Multivariate Datasets
    """
    def __init__(self, series, label, device, window_size=120, stride=1):
        super(MuliTsBatchedWindowDataset, self).__init__()
        self.series = series
        self.label = label
        self.window_size = window_size
        self.stride = stride
        self.device = device

        if len(self.series.shape) == 1:
            raise ValueError('The `series` must be an Multi array!')

        if label is not None and (label.shape[0] != series.shape[0]):
            raise ValueError('The shape of `label` must agrees with the shape of `series`!')

        # Divide data into several windows
        self.tails = np.arange(window_size, series.shape[0] + 1, stride)

    def __getitem__(self, idx):
        """
            Define data Iterator data generation strategy
            :param idx: Window index
            :return: x:multi_series of a window_size, y_pred: after this window,the next predict_step window , also multi_series
        """
        # Data in the window
        x = self.series[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)

        if self.label is None:
            # Only data
            # return torch.from_numpy(x)
            return torch.from_numpy(x)

        elif self.label is not None:
            # Data and label
            y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        """
            Number of windows
        """
        return self.tails.shape[0]


def create_data_loaders(train_dataset, batch_size, val_split=0.0, shuffle=True, test_dataset=None):
    """
        Create data Iterator
    """
    train_loader, val_loader, test_loader = None, None, None

    # Do not divide data
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,drop_last=True)

    # divide data
    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, pin_memory=True, num_workers=4)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True, pin_memory=True, num_workers=4)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader