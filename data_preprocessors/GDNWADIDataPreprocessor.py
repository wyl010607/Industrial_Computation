import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
import torch
from torch.utils.data import random_split, Subset
from .abs import AbstractDataPreprocessor
from utils.scaler import StandardScaler, MinMaxScaler
from models.public.GDN.net_struct import NetStruc
from datasets.TimeDataset import TimeDataset


class GDNWDataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for DCS data.
    """

    def __init__(
        self,
        data_train_path,
        data_test_path,
        train,
        test,
        train_df,
        test_df,
        labels,
        down_ratio=10,
        *args,
        **kwargs
    ):
        """
        Initialize the SWaT Data Preprocessor.
        """

        super().__init__(data_train_path, data_test_path, *args, **kwargs)
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path
        self.update_dataset_params = {}
        self.update_model_params = {}
        self.update_trainer_params = {}
        self.train_df = train_df
        self.test_df = test_df
        self.train = train
        self.test = test
        self.labels = labels
        self.down_ratio = down_ratio
        self.update_model_params["down_ration"] = self.down_ratio
        self.kwargs = kwargs

    def downsample(self, data, labels, down_len):
        np_data = np.array(data)
        np_labels = np.array(labels)
        orig_len, col_num = np_data.shape
        down_time_len = orig_len // down_len
        np_data = np_data.transpose()
        d_data = np_data[:, :down_time_len * down_len].reshape(col_num, -1, down_len)
        d_data = np.median(d_data, axis=2).reshape(col_num, -1)
        d_labels = np_labels[:down_time_len * down_len].reshape(-1, down_len)
        d_labels = np.round(np.max(d_labels, axis=1))
        d_data = d_data.transpose()

        return d_data.tolist(), d_labels.tolist()

    def get_valid(self, train_dataset, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)
        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)
        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        return train_subset, val_subset

    def load_data(self):
        """
        Load data from the specified file path.

        This method loads the data, time stamps, and variable index dictionary
        from a file at `self.data_path`.
        """
        self.train = pd.read_csv(self.data_train_path, index_col=0)
        self.test = pd.read_csv(self.data_test_path, index_col=0)

    def preprocess(self):
        """
        Preprocess SWaT data.

        依次执行数据类型转换(Tofloat64、最小最大归一化与滑动窗口划分)
        Returns
        -------
        np.ndarray
            The preprocessed data array.
        """
        self.train = self.train.iloc[:, 2:]
        self.test = self.test.iloc[:, 3:]

        self.train = self.train.fillna(self.train.mean())
        self.test = self.test.fillna(self.test.mean())
        self.train = self.train.fillna(0)
        self.test = self.test.fillna(0)
        # trim column names
        self.train = self.train.rename(columns=lambda x: x.strip())
        self.test = self.test.rename(columns=lambda x: x.strip())
        train_labels = self.train.attack
        test_labels = self.test.attack
        self.test = self.test.drop(columns=['attack'])
        '''
        scaler = MinMaxScaler(axis=0)
        x_train, x_test = scaler.transform(scaler.fit(self.train.values)), scaler.transform(scaler.fit(self.test.values))
        '''
        for i, col in enumerate(self.train.columns):
            self.train.loc[:, col] = self.train[:, i]
            self.test.loc[:, col] = self.test[:, i]
        # 下采样
        d_train_x, d_train_labels = self.downsample(self.train.values, train_labels, self.down_ratio)
        d_test_x, d_test_labels = self.downsample(self.test.values, test_labels, self.down_ratio)
        train_df = pd.DataFrame(d_train_x, columns=self.train.columns)
        test_df = pd.DataFrame(d_test_x, columns=self.test.columns)
        test_df['attack'] = d_test_labels
        train_df = train_df.iloc[2160:]

        return train_df, test_df

    def split_data(self, preprocessed_data):
        """
        Split the preprocessed data into training, validation, and testing sets.

        在时间序列异常检测任务中，需要分别按照训练集和测试集读取数据，在split_data模块需划分
        训练集与验证集的滑动窗口

        Parameters
        ----------
        preprocessed_data : np.ndarray
            The preprocessed data array to be split.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing the training, validation, and test data arrays, respectively.
        """
        self.train_df, self.test_df = preprocessed_data
        list_path = self.kwargs.get('list_path')
        net = NetStruc('wadi', list_path, self.train_df)
        feature_map = net.get_feature()
        fc_edge_index = net.graph_struct()
        self.train_df, self.test_df = net.construct_data(self.train_df, feature_map, labels=0), net.construct_data(
            self.test_df, feature_map, labels=0)
        self.train_df = TimeDataset(self.train_df, fc_edge_index, mode='train', slide_window=5, slide_stride=1)
        self.train_df, val_df = self.get_valid(self.train_df)
        return self.train_df, val_df, self.test_df
