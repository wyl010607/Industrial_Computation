import numpy as np
import pandas as pd
from sklearn import preprocessing
from .abs import AbstractDataPreprocessor
from utils.scaler import StandardScaler, MinMaxScaler


class DCDataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for DCS data.
    """

    def __init__(
        self,
        data_train_path_s,
        data_test_path_s,
        data_train_path_w,
        data_test_path_w,
        train,
        val,
        test,
        labels,
        index,
        window=100,
        step=1,
        ignore=None,
        flag="train",
        dataset="swat",
        *args,
        **kwargs
    ):
        super().__init__(data_train_path_s, data_test_path_s, data_train_path_w, data_test_path_w, *args, **kwargs)
        self.data_train_path_s = data_train_path_s
        self.data_test_path_s = data_test_path_s
        self.data_train_path_w = data_train_path_w
        self.data_test_path_w = data_test_path_w
        self.data_train_path = data_train_path_s
        self.data_test_path = data_test_path_s
        self.update_dataset_params = {}
        self.update_model_params = {}
        self.update_trainer_params = {}
        self.train = train
        self.val = val
        self.test = test
        self.labels = labels
        self.flag = flag
        self.step = step
        self.window = window
        self.update_dataset_params["window"] = self.window
        self.index = index
        self.dataset = dataset
        self.ignore = ignore

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.window) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.window) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.window) // self.step + 1
        else:
            return (self.test.shape[0] - self.window) // self.window + 1

    def load_data(self):
        """
        Load data from the specified file path.

        This method loads the data, time stamps, and variable index dictionary
        from a file at `self.data_path`.
        """
        if self.dataset=='swat':
            self.data_train_path = self.data_train_path_s
            self.data_test_path = self.data_test_path_s
        if self.dataset=='wadi':
            self.data_train_path = self.data_train_path_w
            self.data_test_path = self.data_test_path_w
        self.train = pd.read_csv(self.data_train_path)
        self.train = self.train.values[:, :-1]
        self.test = pd.read_csv(self.data_test_path)
        self.labels = self.test.values[:, -1:]
        self.test = self.test.values[:, :-1]

    def preprocess(self):
        """
        Returns
        -------
        np.ndarray
            The preprocessed data array.
        """
        # Normalization
        #scaler = preprocessing.StandardScaler()
        '''
        scaler = StandardScaler(axis=0)
        x_n, x_a = self.train.values, self.test.values
        self.train, self.test = scaler.transform(scaler.fit(x_n)), scaler.transform(scaler.fit(x_a))
        '''
        if self.dataset == 'swat':
            self.ignore = (10,)
        if self.dataset == 'wadi':
            self.ignore = (102,)
        if self.ignore is not None:
            self.train = np.delete(self.train, self.ignore, axis=1)
            self.test = np.delete(self.test, self.ignore, axis=1)
        return self.train,self.test,self.labels

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
        self.train,self.test,self.labels = preprocessed_data
        data_len = len(self.train)
        if self.dataset == 'swat':
            self.val = self.train[(int)(data_len * 0.8):]
        if self.dataset == 'wadi':
            self.val = self.test
        '''
        self.index = self.index * self.step
        train_w = np.float32(self.train[self.index:self.index + self.window]), np.float32(self.labels[0:self.window])
        val_w = np.float32(self.val[self.index:self.index + self.window]), np.float32(self.labels[0:self.window])
        test_w = np.float32(self.test[self.index:self.index + self.window]), np.float32(
            self.labels[self.index:self.index + self.window])
        '''
        train_window = []
        val_window = []
        test_window = []
        labels_window = []
        for i in range(0, len(self.labels) - self.window + 1, self.step):
            labels_window.append(self.labels[i:i + self.window])
        label_w = np.array(labels_window, dtype=np.float32)
        for i in range(0, len(self.train) - self.window + 1, self.step):
            train_window.append(self.train[i:i + self.window])
        train_w = np.array(train_window, dtype=np.float32)
        for i in range(0, len(self.val) - self.window + 1, self.step):
            val_window.append(self.val[i:i + self.window])
        val_w = np.array(val_window, dtype=np.float32)
        for i in range(0, len(self.test) - self.window + 1, self.step):
            test_window.append(self.test[i:i + self.window])
        test_w = np.array(test_window, dtype=np.float32)
        train_w = np.array(train_w, label_w)
        val_w = np.array(val_w, label_w)
        test_w = np.array(test_w, label_w)

        return train_w, val_w, test_w
