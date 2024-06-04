import numpy as np
import pandas as pd
from .abs import AbstractDataPreprocessor
import torch
from sklearn import preprocessing



class GDNDataPreprocessor(AbstractDataPreprocessor):
    def __init__(
        self,
        data_train_path,
        data_test_path,
        dataset,
        window_size,
        hidden_size,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.train_path = data_train_path
        self.test_path = data_test_path
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.dataset = dataset
        self.update_dataset_params = {}
        self.update_model_params = {}
        self.update_trainer_params = {}

    def load_data(self):
        normal = pd.read_csv(self.train_path)  # , nrows=1000)
        attack = pd.read_csv(self.test_path, sep=";")  # , nrows=1000)

    def preprocess(self):
        """
                Preprocess SWaT & WADI data.

                依次执行数据类型转换(Tofloat64、最小最大归一化与滑动窗口划分)
                Returns
                -------
                np.ndarray
                    The preprocessed data array.
                """
        if self.dataset=='swat':
            normal = pd.read_csv(self.train_path)  # , nrows=1000)
            normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)

            attack = pd.read_csv(self.test_path, sep=";")  # , nrows=1000)
            attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)
            labels = [float(label != 'Normal') for label in attack["Normal/Attack"].values]

            # 数据类型转换
            for i in list(normal):
                normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
            normal = normal.astype(float)
            for i in list(attack):
                attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
            attack = attack.astype(float)

            min_max_scaler = preprocessing.MinMaxScaler()
            x = normal.values
            x_scaled = min_max_scaler.fit_transform(x)
            process_normal = pd.DataFrame(x_scaled)
            xa = attack.values
            xa_scaled = min_max_scaler.transform(xa)
            process_attack = pd.DataFrame(xa_scaled)

            windows_normal = process_normal.values[
                np.arange(self.window_size)[None, :] + np.arange(process_normal.shape[0] - self.window_size)[:, None]]
            windows_attack = process_attack.values[
                np.arange(self.window_size)[None, :] + np.arange(process_attack.shape[0] - self.window_size)[:, None]]
            preprocessed_data = windows_normal, windows_attack, labels

            return preprocessed_data

        if self.dataset=='wadi':
            normal = pd.read_csv(self.train_path)  # , nrows=1000)
        # 去除训练数据集中的空属性与时间属性
            normal = normal.drop(normal.columns[[0, 1, 2, 50, 51, 86, 87]], axis=1)
            normal = normal.groupby(np.arange(len(normal.index)) // 5).mean()
            normal = normal.astype(float)

            min_max_scaler = preprocessing.MinMaxScaler()
            x = normal.values
            x_scaled = min_max_scaler.fit_transform(x)
            normal = pd.DataFrame(x_scaled)

            nanv = normal.isnull().sum().sum()
            process_normal = normal.fillna(0)

            attack = pd.read_csv(self.test_path, sep=";")  # , nrows=1000)
            labels = []

            attack.reset_index()
            for index, row in attack.iterrows():
                date_temp = row['Date']
                date_mask = "%m/%d/%Y"
                date_obj = datetime.strptime(date_temp, date_mask)
                time_temp = row['Time']
                time_mask = "%I:%M:%S.%f %p"
                time_obj = datetime.strptime(time_temp, time_mask)

                if date_obj == datetime.strptime('10/9/2017', '%m/%d/%Y'):
                    if time_obj >= datetime.strptime('7:25:00.000 PM',
                                                     '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                            '7:50:16.000 PM', '%I:%M:%S.%f %p'):
                        labels.append('Attack')
                        continue

                if date_obj == datetime.strptime('10/10/2017', '%m/%d/%Y'):
                    if time_obj >= datetime.strptime('10:24:10.000 AM',
                                                     '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
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
                                                       '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                            '1:50:40.000 PM', '%I:%M:%S.%f %p'):
                        labels.append('Attack')
                        continue
                    elif time_obj >= datetime.strptime('2:48:17.000 PM',
                                                       '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                            '2:59:55.000 PM', '%I:%M:%S.%f %p'):
                        labels.append('Attack')
                        continue
                    elif time_obj >= datetime.strptime('5:40:00.000 PM',
                                                       '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                            '5:49:40.000 PM', '%I:%M:%S.%f %p'):
                        labels.append('Attack')
                        continue
                    elif time_obj >= datetime.strptime('10:55:00.000 AM',
                                                       '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                            '10:56:27.000 AM', '%I:%M:%S.%f %p'):
                        labels.append('Attack')
                        continue

                if date_obj == datetime.strptime('10/11/2017', '%m/%d/%Y'):
                    if time_obj >= datetime.strptime('11:17:54.000 AM',
                                                     '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
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
                                                       '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                            '3:37:00.000 PM', '%I:%M:%S.%f %p'):
                        labels.append('Attack')
                        continue
                labels.append('Normal')

            attack = attack.drop(attack.columns[[0, 1, 2, 50, 51, 86, 87]],
                                 axis=1)  # Drop the empty and date/time columns

            # Downsampling the attack data
            attack = attack.groupby(np.arange(len(attack.index)) // 5).mean()
            labels_down = []
            for i in range(len(labels) // 5):
                if labels[5 * i:5 * (i + 1)].count('Attack'):
                    labels_down.append(1.0)  # Attack
                else:
                    labels_down.append(0.0)  # Normal

            if labels[5 * (i + 1):].count('Attack'):
                labels_down.append(1.0)  # Attack
            else:
                labels_down.append(0.0)  # Normal
            attack = attack.astype(float)
            xa = attack.values
            xa_scaled = min_max_scaler.transform(xa)
            process_attack = pd.DataFrame(xa_scaled)

            windows_normal = normal.values[
                np.arange(self.window_size)[None, :] + np.arange(process_normal.shape[0] - self.window_size)[:, None]]
            windows_attack = attack.values[
                np.arange(self.window_size)[None, :] + np.arange(process_attack.shape[0] - self.window_size)[:, None]]

            preprocessed_data = windows_normal, windows_attack, labels, labels_down

            return preprocessed_data


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
        if self.dataset=='swat':
            windows_normal, windows_attack, labels = preprocessed_data
            w_size = windows_normal.shape[1] * windows_normal.shape[2]
            z_size = windows_normal.shape[1] * self.hidden_size

            windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
            windows_normal_val = windows_normal[
                int(np.floor(.8 * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

            windows_labels = []
            for i in range(len(labels) - self.window_size):
                windows_labels.append(list(np.int_(labels[i:i + self.window_size])))

            return windows_normal_train, windows_normal_val, windows_attack, windows_labels

        if self.dataset=='wadi':
            windows_normal, windows_attack, labels, labels_down = preprocessed_data
            w_size = windows_normal.shape[1] * windows_normal.shape[2]
            z_size = windows_normal.shape[1] * self.hidden_size
            windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
            windows_normal_val = windows_normal[
                                 int(np.floor(.8 * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]
            windows_labels = []
            for i in range(len(labels_down) - self.window_size):
                windows_labels.append(list(np.int_(labels_down[i:i + self.window_size])))

            return windows_normal_train, windows_normal_val, windows_attack, windows_labels
