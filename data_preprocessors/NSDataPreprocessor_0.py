import datetime

import numpy as np
import torch

from .abs import AbstractDataPreprocessor
import pandas as pd
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
from utils.timefeatures import time_features
from utils.tools import StandardScaler



class NSDataPreprocessor_0(AbstractDataPreprocessor):
    """
    Data Preprocessor for DCS data.
    """

    def __init__(
        self,
        data_path,
        process_vars_list=None,
        control_vars_list=None,
        disturb_vars_list=None,
        *args,
        **kwargs
    ):

        super().__init__(data_path, *args, **kwargs)
        self.data = None
        self.adj_mx = None
        self.vars_index_dict = None
        self.time_stamp_array = None
        self.history_len = kwargs.get("history_len")
        self.forecast_len = kwargs.get("forecast_len")
        self.timeenc = kwargs.get("timeenc")
        self.freq = kwargs.get("freq")
        self.label_len=kwargs.get("label_len")
        self.features = kwargs.get("features")
        self.process_vars_list = (
            process_vars_list if process_vars_list is not None else []
        )
        self.control_vars_list = (
            control_vars_list if control_vars_list is not None else []
        )
        self.disturb_vars_list = (
            disturb_vars_list if disturb_vars_list is not None else []
        )
        self.load_data()


    def load_data(self):

        data = np.load(self.data_path, allow_pickle=True)
        df_data, df_stamp, self.vars_index_dict = (
            pd.DataFrame(data["data_array"][:1000]),
            data["time_stamp_array"][:1000].tolist(),
            data["vars_index_dict"].tolist(),
        )
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_vali = len(df_data) - num_train - num_test
        border1s = [0, num_train - self.history_len, len(df_data) - num_test - self.history_len]
        border2s = [num_train, num_train + num_vali, len(df_data)]
        '''
        self.scaler = StandardScaler()
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        self.data = self.scaler.transform(df_data.values)
        '''

        self.data=df_data.values

        df_stamp = pd.DataFrame(df_stamp, columns=['date'])
        def timestamp2datetime(timestamp):
            return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        df_stamp = pd.DataFrame(df_stamp['date'].map(timestamp2datetime))

        '''
        train_df_stamp = df_stamp[['date']][border1s[0]:border2s[0]]
        train_df_stamp['date'] = pd.to_datetime(train_df_stamp.date)
        train_date_stamp = time_features(train_df_stamp)
        date_scaler = sklearn_StandardScaler().fit(train_date_stamp)


        data_stamp = []
        for i in range(3):
            df_stamp_temp = df_stamp[['date']][border1s[i]:border2s[i]]
            df_stamp_temp['date'] = pd.to_datetime(df_stamp_temp.date)
            data_stamp_temp = time_features(df_stamp_temp, freq=self.freq)
            data_stamp_temp = date_scaler.transform(data_stamp_temp)
            data_stamp.append(data_stamp_temp)

        self.data_stamp = data_stamp
         '''
        data_stamp=[]
        for i in range(3):
            df_stamp_temp=df_stamp[['date']][border1s[i]:border2s[i]]
            df_stamp_temp['date'] = pd.to_datetime(df_stamp_temp.date)
            if self.timeenc == 0:
                df_stamp_temp['month'] = df_stamp_temp.date.apply(lambda row: row.month, 1)
                df_stamp_temp['day'] = df_stamp_temp.date.apply(lambda row: row.day, 1)
                df_stamp_temp['weekday'] = df_stamp_temp.date.apply(lambda row: row.weekday(), 1)
                df_stamp_temp['hour'] = df_stamp_temp.date.apply(lambda row: row.hour, 1)
                data_stamp.append(df_stamp_temp.drop(['date'], 1).values)
            elif self.timeenc == 1:
                data_stamp_0 = time_features(pd.to_datetime(df_stamp_temp['date'].values), freq=self.freq)
                data_stamp_0 = data_stamp_0.transpose(1, 0)
                data_stamp.append(data_stamp_0)

        self.data_stamp=data_stamp

    def preprocess(self):

        self.update_dataset_params["history_len"] = self.history_len
        self.update_dataset_params["forecast_len"] = self.forecast_len
        self.update_dataset_params["label_len"] = self.label_len
        self.update_dataset_params["data_stamp"] = self.data_stamp
        self.update_model_params["history_len"] = self.history_len
        self.update_model_params["label_len"] = self.label_len
        self.update_model_params["forecast_len"] = self.forecast_len
        self.update_model_params["freq"] = self.freq
        self.update_trainer_params["label_len"] = self.label_len
        self.update_trainer_params["forecast_len"] = self.forecast_len
        self.update_trainer_params["features"] = self.features
        indices = [
            self.vars_index_dict[var]
            for var in (
                self.process_vars_list + self.control_vars_list + self.disturb_vars_list
            )
        ]
        preprocessed_data = self.data[:, indices]

        self.update_trainer_params = {
            "PV_index_list": [
                self.vars_index_dict[var] for var in self.process_vars_list
            ],
            "OP_index_list": [
                self.vars_index_dict[var] for var in self.control_vars_list
            ],
            "DV_index_list": [
                self.vars_index_dict[var] for var in self.disturb_vars_list
            ],
        }

        # load adjacency matrix
        self.update_dataset_params["history_len"] = self.history_len
        self.update_dataset_params["forecast_len"] = self.forecast_len
        self.update_dataset_params["data_stamp"] = self.data_stamp

        self.update_trainer_params["forecast_len"] = self.forecast_len

        self.update_model_params["forecast_len"] = self.forecast_len
        return preprocessed_data

    def split_data(self, preprocessed_data):

        num_train = int(len(preprocessed_data) * 0.7)
        num_test = int(len(preprocessed_data) * 0.2)
        num_vali = len(preprocessed_data) - num_train - num_test
        border1s = [0, num_train - self.history_len, len(preprocessed_data) - num_test - self.history_len]
        border2s = [num_train, num_train + num_vali, len(preprocessed_data)]
        train_data = preprocessed_data[:border2s[0]]
        test_data = preprocessed_data[border1s[2]:border2s[2]]
        valid_data = preprocessed_data[border1s[1]:border2s[1]]
        return train_data, valid_data, test_data
