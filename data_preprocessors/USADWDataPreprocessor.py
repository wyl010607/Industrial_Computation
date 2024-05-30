import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from .abs import AbstractDataPreprocessor
from utils.scaler import StandardScaler, MinMaxScaler


class USADWDataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for WADI data.
    """

    def __init__(
            self,
            data_train_path,
            data_test_path,
            normal=None,
            attack=None,
            labels=None,
            windows_normal=None,
            windows_attack=None,
            window_size=10,
            w_size=None,
            z_size=None,
            *args,
            **kwargs
    ):
        """
        Initialize the WADI Data Preprocessor.
        """

        super().__init__(data_train_path, data_test_path, *args, **kwargs)
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path
        self.hidden_size = kwargs.get("hidden_size")
        self.update_dataset_params = {}
        self.update_model_params = {}
        self.update_trainer_params = {}
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
        self.normal = pd.read_csv(self.data_train_path, sep=',', skiprows=[0, 1, 2, 3],
                             skip_blank_lines=True)  # , nrows=1000)
        # 去除训练数据集中的空属性与时间属性
        self.normal = self.normal.drop(self.normal.columns[[0, 1, 2, 50, 51, 86, 87]], axis=1)

        self.attack = pd.read_csv(self.data_test_path, sep=";")  # , nrows=1000)
        self.attack=self.attack.drop(self.attack.columns[[0,1,2,50,51,86,87]],axis=1)
        labels = []
        self.attack.reset_index()
        for index, row in self.attack.iterrows():
            date_temp = row['Date']
            date_mask = "%m/%d/%Y"
            date_obj = datetime.strptime(date_temp, date_mask)
            time_temp = row['Time']
            time_mask = "%I:%M:%S.%f %p"
            time_obj = datetime.strptime(time_temp, time_mask)
            if date_obj == datetime.strptime('10/9/2017', '%m/%d/%Y'):
                if time_obj >= datetime.strptime('7:25:00.000 PM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '7:50:16.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
            if date_obj == datetime.strptime('10/10/2017', '%m/%d/%Y'):
                if time_obj >= datetime.strptime('10:24:10.000 AM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '10:34:00.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('10:55:00.000 AM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '11:24:00.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('11:30:40.000 AM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '11:44:50.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('1:39:30.000 PM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime('1:50:40.000 PM',
                                                                                                       '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('2:48:17.000 PM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime('2:59:55.000 PM',
                                                                                                       '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('5:40:00.000 PM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime('5:49:40.000 PM',
                                                                                                       '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('10:55:00.000 AM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '10:56:27.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
            if date_obj == datetime.strptime('10/11/2017', '%m/%d/%Y'):
                if time_obj >= datetime.strptime('11:17:54.000 AM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '11:31:20.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('11:36:31.000 AM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '11:47:00.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('11:59:00.000 AM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '12:05:00.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('12:07:30.000 PM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '12:10:52.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('12:16:00.000 PM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                        '12:25:36.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj >= datetime.strptime('3:26:30.000 PM',
                                                   '%I:%M:%S.%f %p') and time_obj <= datetime.strptime('3:37:00.000 PM',
                                                                                                       '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
            labels.append('Normal')

    def preprocess(self):
        """
        Preprocess the loaded data.

        This method extracts specific indices from the data based on the process,
        control, and disturb variable lists, and prepares it for further processing.
        by the way, load the adj_mx and add it to the update_model_params

        Returns
        -------
        np.ndarray
            The preprocessed data array.
        """
        # 数据集下采样
        self.normal = self.normal.groupby(np.arange(len(self.normal.index)) // 5).mean()
        self.attack = self.attack.groupby(np.arange(len(self.attack.index)) // 5).mean()
        self.normal = self.normal.astype(float)
        self.attack = self.attack.astype(float)

        # Normalization
        # scaler = preprocessing.StandardScaler()
        '''
        scaler = MinMaxScaler(axis=0)
        x_n, x_a = self.normal.values, self.attack.values
        x_nn, x_an = scaler.transform(scaler.fit(x_n)), scaler.transform(scaler.fit(x_a))
        '''
        process_normal, process_attack = pd.DataFrame(self.normal), pd.DataFrame(self.attack)
        # 缺失值补全
        process_normal = process_normal.fillna(0)

        self.update_dataset_params["window"] = self.window_size
        self.windows_normal = process_normal.values[
            np.arange(self.window_size)[None, :] + np.arange(process_normal.shape[0] - self.window_size)[:, None]]
        self.windows_attack = process_attack.values[
            np.arange(self.window_size)[None, :] + np.arange(process_attack.shape[0] - self.window_size)[:, None]]

        return self.windows_normal, self.windows_attack

    def split_data(self, preprocessed_data):
        """
        Split the preprocessed data into training, validation, and testing sets.

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
                             int(np.floor(.8 * self.windows_normal.shape[0])):int(
                                 np.floor(self.windows_normal.shape[0]))]

        return windows_normal_train, windows_normal_val, self.windows_attack

def label_len(self):
    len_l = len(self.labels)
    return self.labels, len_l
