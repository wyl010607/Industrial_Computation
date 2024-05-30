import numpy as np
import pandas as pd
from sklearn import preprocessing
from .abs import AbstractDataPreprocessor
from utils.scaler import StandardScaler, MinMaxScaler


class USADSDataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for DCS data.
    """

    def __init__(
        self,
        data_train_path,
        data_test_path,
        normal=None,
        attack=None,
        labels=None,
        windows_normal =None,
        windows_attack = None,
        window_size=10,
        w_size=None,
        z_size=None,
        *args,
        **kwargs
    ):
        """
        Initialize the SWaT Data Preprocessor.
        """

        super().__init__(data_train_path, data_test_path, *args, **kwargs)
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path
        self.hidden_size = 40
        self.update_dataset_params = {}
        self.update_model_params = {}
        self.update_trainer_params = {}
        self.update_model_params["hidden_size"] = self.hidden_size
        self.normal = normal
        self.attack = attack
        self.labels = labels
        self.windows_normal = windows_normal
        self.windows_attack = windows_attack
        self.window_size = window_size
        self.w_size = w_size
        self.z_size = z_size

    def load_data(self):
        """
        Load data from the specified file path.

        This method loads the data, time stamps, and variable index dictionary
        from a file at `self.data_path`.
        """
        self.normal = pd.read_csv(self.data_train_path)  # , nrows=1000)
        self.normal = self.normal.drop(["Timestamp", "Normal/Attack"], axis=1)

        self.attack = pd.read_csv(self.data_test_path, sep=";")  # , nrows=1000)
        self.attack = self.attack.drop(["Timestamp", "Normal/Attack"], axis=1)
        self.labels = [float(label != 'Normal') for label in self.attack["Normal/Attack"].values]

    def preprocess(self):
        """
        Preprocess SWaT data.

        依次执行数据类型转换(Tofloat64、最小最大归一化与滑动窗口划分)
        Returns
        -------
        np.ndarray
            The preprocessed data array.
        """
        # 数据类型转换
        for i in list(self.normal):
            self.normal[i] = self.normal[i].apply(lambda x: str(x).replace(",", "."))
        self.normal = self.normal.astype(float)
        for i in list(self.attack):
            self.attack[i] = self.attack[i].apply(lambda x: str(x).replace(",", "."))
        self.attack = self.attack.astype(float)

        # Normalization
        #scaler = preprocessing.StandardScaler()
        '''
        scaler = MinMaxScaler(axis=0)
        x_n, x_a = self.normal.values, self.attack.values
        x_nn, x_an = scaler.transform(scaler.fit(x_n)), scaler.transform(scaler.fit(x_a))
        '''
        process_normal, process_attack = pd.DataFrame(self.normal), pd.DataFrame(self.attack)

        self.update_dataset_params["window"] = self.window_size
        self.update_trainer_params["window"] = self.window_size
        self.windows_normal = process_normal.values[
            np.arange(self.window_size)[None, :] + np.arange(process_normal.shape[0] - self.window_size)[:, None]]
        self.windows_attack = process_attack.values[
            np.arange(self.window_size)[None, :] + np.arange(process_attack.shape[0] - self.window_size)[:, None]]

        return self.windows_normal, self.windows_attack

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
        self.windows_normal, self.windows_attack = preprocessed_data
        self.w_size = self.windows_normal.shape[1] * self.windows_normal.shape[2]
        self.z_size = self.windows_normal.shape[1] * self.hidden_size

        self.update_dataset_params["w_size"] = self.w_size
        self.update_dataset_params["z_size"] = self.z_size
        self.update_model_params["w_size"] = self.w_size
        self.update_model_params["z_size"] = self.z_size

        windows_normal_train = self.windows_normal[:int(np.floor(.8 * self.windows_normal.shape[0]))]
        windows_normal_val = self.windows_normal[
                             int(np.floor(.8 * self.windows_normal.shape[0])):int(np.floor(self.windows_normal.shape[0]))]

        return windows_normal_train, windows_normal_val, self.windows_attack


def label_len(self):
    len_l = len(self.labels)
    return self.labels, len_l
