
from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from .abs import AbstractDataPreprocessor
warnings.filterwarnings('ignore')

class NSDataPreprocessor(AbstractDataPreprocessor):


    def __init__(self,data_path,*args, **kwargs):

        super().__init__(data_path, *args, **kwargs)
        self.data_path = data_path
        self.update_dataset_params = {}
        self.update_model_params = {}
        self.update_trainer_params = {}
        self.data=None
        self.target = kwargs.get("target")
        self.features = kwargs.get("features")
        self.timeenc = kwargs.get("timeenc")
        self.freq = kwargs.get("freq")
        self.history_len = kwargs.get("history_len")
        self.forecast_len = kwargs.get("forecast_len")
        self.label_len = kwargs.get("label_len")
        self.load_data()

    def load_data(self):
        """
        Abstract method to load data.
        """
        self.scaler = StandardScaler()
        self.data= pd.read_csv(os.path.join(self.data_path))


        cols = list(self.data.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = self.data[['date'] + cols + [self.target]]
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.history_len, len(df_raw) - num_test - self.history_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        #train_data = df_data[border1s[0]:border2s[0]]
        #self.scaler.fit(train_data.values)
        #df_data = self.scaler.transform(df_data.values)
        self.df_data=df_data.values

        data_stamp=[]
        for i in range(3):
            df_stamp=df_raw[['date']][border1s[i]:border2s[i]]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp.append(df_stamp.drop(['date'], 1).values)

            elif self.timeenc == 1:
                data_stamp_0 = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp_0 = data_stamp_0.transpose(1, 0)
                data_stamp.append(data_stamp_0)

        self.data_stamp=data_stamp







    def preprocess(self):


        self.update_dataset_params["history_len"] = self.history_len
        self.update_dataset_params["label_len"] = self.label_len
        self.update_dataset_params["forecast_len"] = self.forecast_len
        self.update_dataset_params["data_stamp"] = self.data_stamp
        self.update_model_params["history_len"] = self.history_len
        self.update_model_params["label_len"] = self.label_len
        self.update_model_params["freq"] = self.freq
        self.update_model_params["forecast_len"] = self.forecast_len
        self.update_trainer_params["label_len"] = self.label_len
        self.update_trainer_params["forecast_len"] = self.forecast_len
        self.update_trainer_params["features"] = self.features
        return self.df_data



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
