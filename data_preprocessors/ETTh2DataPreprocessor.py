import numpy as np
import torch
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
from .abs import AbstractDataPreprocessor
import pandas as pd
from utils.timefeatures import time_features
from utils.tools import StandardScaler

class ETTh2DataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for DCS data.
    """

    def __init__(
        self,
        data_path,
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

        self.load_data()


    def load_data(self):
        """
        Load data from the specified file path.

        This method loads the data, time stamps, and variable index dictionary
        from a file at `self.data_path`.
        """
        border1s = [0, 4 * 30 * 24 - self.history_len, 5 * 30 * 24 - self.history_len]
        border2s = [4 * 30 * 24, 5 * 30 * 24, 20 * 30 * 24]
        df_raw = pd.read_csv(self.data_path)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        self.scaler = StandardScaler()
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        self.data = self.scaler.transform(df_data.values)


        train_df_stamp = df_raw[['date']][border1s[0]:border2s[0]]

        train_df_stamp['date'] = pd.to_datetime(train_df_stamp.date)
        train_date_stamp = time_features(train_df_stamp, timeenc=self.timeenc)
        date_scaler = sklearn_StandardScaler().fit(train_date_stamp)

        data_stamp =[]
        for i in range(3):
            df_stamp = df_raw[['date']] [border1s[i]:border2s[i]]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            data_stamp_temp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
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
        self.update_dataset_params["data_stamp"] = self.data_stamp

        self.update_trainer_params["forecast_len"] = self.forecast_len

        self.update_model_params["forecast_len"] = self.forecast_len
        return self.data

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
        '''total_length = len(preprocessed_data)
        train_end = int(self.train_ratio * total_length)
        valid_end = int((self.train_ratio + self.valid_ratio) * total_length)

        train_data = preprocessed_data[:train_end]
        valid_data = preprocessed_data[train_end:valid_end]
        test_data = preprocessed_data[valid_end:]'''

        train_data = preprocessed_data[: 4*30*24]
        valid_data = preprocessed_data[4*30*24 - self.history_len:5*30*24]
        test_data = preprocessed_data[5*30*24 - self.history_len:20*30*24]

        return train_data, valid_data, test_data
