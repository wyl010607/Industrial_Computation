import datetime

import numpy as np
import torch

from .abs import AbstractDataPreprocessor
import pandas as pd
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
from utils.timefeatures import time_features
from utils.tools import StandardScaler



class CFSNETDataPreprocessor(AbstractDataPreprocessor):
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
        """
        Initialize the DCS Data Preprocessor.

        Parameters
        ----------
        data_path : str
            Path to the data file.
        adj_mx_path : str, optional
            Path to the adjacency matrix. Default is None.
        train_ratio : float, optional
            Ratio of training data. Default is 0.6.
        valid_ratio : float, optional
            Ratio of validation data. Default is 0.2.
        process_vars_list : list of str, optional
            List of process variable names. Default is an empty list.
        control_vars_list : list of str, optional
            List of control variable names. Default is an empty list.
        disturb_vars_list : list of str, optional
            List of disturbance variable names. Default is an empty list.
        """

        super().__init__(data_path, *args, **kwargs)
        self.data = None
        self.adj_mx = None
        self.vars_index_dict = None
        self.time_stamp_array = None
        self.history_len = kwargs.get("history_len")
        self.forecast_len = kwargs.get("forecast_len")
        self.timeenc = kwargs.get("timeenc")
        self.freq = kwargs.get("freq")

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
        """
        Load data from the specified file path.

        This method loads the data, time stamps, and variable index dictionary
        from a file at `self.data_path`.
        """
        data = np.load(self.data_path, allow_pickle=True)
        df_data, df_stamp, self.vars_index_dict = (
            pd.DataFrame(data["data_array"]),
            data["time_stamp_array"].tolist(),
            data["vars_index_dict"].tolist(),
        )

        '''
        self.scaler = StandardScaler()
        train_data = df_data[:int(self.train_ratio * len(df_data))]
        self.scaler.fit(train_data.values)
        self.data = self.scaler.transform(df_data.values)
        '''
        self.data = df_data.values
        df_stamp = pd.DataFrame(df_stamp, columns=['date'])
        def timestamp2datetime(timestamp):
            return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        df_stamp = pd.DataFrame(df_stamp['date'].map(timestamp2datetime))

        total_length = len(df_data)
        border1s = [0, int(self.train_ratio * total_length) - self.history_len, int((self.train_ratio + self.valid_ratio) * total_length) - self.history_len]
        border2s = [int(self.train_ratio * total_length), int((self.train_ratio + self.valid_ratio) * total_length), total_length]

        train_df_stamp = df_stamp[['date']][border1s[0]:border2s[0]]
        train_df_stamp['date'] = pd.to_datetime(train_df_stamp.date)
        train_date_stamp = time_features(train_df_stamp, timeenc=self.timeenc)
        date_scaler = sklearn_StandardScaler().fit(train_date_stamp)


        data_stamp = []
        for i in range(3):
            df_stamp_temp = df_stamp[['date']][border1s[i]:border2s[i]]
            df_stamp_temp['date'] = pd.to_datetime(df_stamp_temp.date)
            data_stamp_temp = time_features(df_stamp_temp, timeenc=self.timeenc, freq=self.freq)
            data_stamp_temp = date_scaler.transform(data_stamp_temp)
            data_stamp.append(data_stamp_temp)

        self.data_stamp = data_stamp

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
        self.update_dataset_params["history_len"] = self.history_len
        self.update_dataset_params["forecast_len"] = self.forecast_len


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
        total_length = len(preprocessed_data)
        train_end = int(self.train_ratio * total_length)
        valid_end = int((self.train_ratio + self.valid_ratio) * total_length)

        train_data = preprocessed_data[:train_end]
        valid_data = preprocessed_data[train_end:valid_end]
        test_data = preprocessed_data[valid_end:]

        return train_data, valid_data, test_data
