import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


def data_name2nc(data_name: str):
    nc = 0
    if data_name == 'SMD':
        nc = 38
    elif data_name == 'SWAT':
        nc = 51 - 1
    elif data_name == 'WADI':
        nc = 127 - 1
    else:
        raise ValueError(f'no this dataset {data_name}')
    return nc


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")[:, :]
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")[:, :]
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")[:]

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size)
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.zeros(self.win_size)
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class WADI_SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", ignore=(102,), scaler=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        data = np.load(data_path + "/WADI_train.npy")
        test_data = np.load(data_path + "/WADI_test.npy")

        if ignore is not None:
            data = np.delete(data, ignore, axis=1)
            test_data = np.delete(test_data, ignore, axis=1)

        if scaler:
            data = self.scaler.fit_transform(data)
            test_data = self.scaler.fit_transform(test_data)

        self.train = data
        self.val = self.test = test_data
        self.test_labels = np.load(data_path + "/WADI_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size)
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.zeros(self.win_size)
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step, flag="train", ignore=(10,), scaler=1):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'SWaT_Dataset_Normal_v1.CSV'))
        test_data = pd.read_csv(os.path.join(root_path, 'SWaT_Dataset_Attack_v0.CSV'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]
        if ignore is not None:
            train_data = np.delete(train_data, ignore, axis=1)
            test_data = np.delete(test_data, ignore, axis=1)

        if scaler:
            train_data = self.scaler.fit_transform(train_data)
            test_data = self.scaler.fit_transform(test_data)

        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size)
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.zeros(self.win_size)
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, win_size=100, step=1, mode='train', dataset='SWAT', dist=0,
                       ret_data=False):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'SWAT'):
        dataset = SWATSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'WADI'):
        dataset = WADI_SegLoader(data_path, win_size, step, mode)
    if ret_data:
        return dataset

    shuffle = False
    if mode == 'train':
        shuffle = True
    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
    else:
        sampler = None

    return dataset, sampler, shuffle
