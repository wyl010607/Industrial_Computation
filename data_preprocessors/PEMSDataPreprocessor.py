import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_preprocessors.abs import AbstractDataPreprocessor


class PEMSDataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for PEMS03, PEMS04, PEMS07, PEMS08 dataset.
    """

    def __init__(
        self,
        data_path,
        steps_per_day=288,
        adj_mx_path=None,
        distance_adj_mx_path=None,
        add_feature_time_of_day=True,
        add_feature_day_of_week=True,
        *args,
        **kwargs
    ):
        """
        Initialize the AirQualityDataPreprocessor.

        Parameters
        ----------
        data_path : str
            Path to the data file. .xlsx file
        adj_mx_path : str, optional
            Path to the adjacency matrix. Default is None.
        distance_adj_mx_path : str, optional
            Path to the distance adjacency matrix. Default is None.
        steps_per_day : int, optional
            Number of steps per day.
        add_feature_time_of_day : bool, optional
            Whether to add the time of day feature. Default is True.
        add_feature_day_of_week : bool, optional
            Whether to add the day of week feature. Default is True.
        """

        super().__init__(data_path, *args, **kwargs)
        self.data = None
        self.datetime_index = None
        self.adj_mx = None
        self.adj_mx_path = adj_mx_path if adj_mx_path is not None else None
        self.distance_adj_mx = None
        self.distance_adj_mx_path = distance_adj_mx_path if distance_adj_mx_path is not None else None
        self.steps_per_day = steps_per_day
        self.history_len = kwargs.get("history_len")
        self.forecast_len = kwargs.get("forecast_len")
        self.add_feature_time_of_day = add_feature_time_of_day
        self.add_feature_day_of_week = add_feature_day_of_week
        self.load_data()

    def load_data(self):
        """
        Load data from the specified file path.
        """
        df = pd.read_hdf(self.data_path)
        self.data = np.expand_dims(df.values, axis=-1)

    def asym_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj).astype(np.float32).todense()

    def preprocess(self):
        """
        Preprocess the loaded data.

        Returns
        -------
        np.ndarray
            The preprocessed data array.
        """
        data = self.data
        l, n, f = data.shape

        # add temporal feature
        feature_list = [data]
        if self.add_feature_time_of_day:
            # numerical time_of_day
            tod = [
                i % self.steps_per_day / self.steps_per_day
                for i in range(data.shape[0])
            ]
            tod = np.array(tod)
            tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(tod_tiled)

        if self.add_feature_day_of_week:
            # numerical day_of_week
            dow = [(i // self.steps_per_day) % 7 / 7 for i in range(data.shape[0])]
            dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(dow_tiled)

        preprocessed_data = np.concatenate(feature_list, axis=-1)
        # load adjacency matrix
        if self.adj_mx_path is not None:
            try:
                with open(self.adj_mx_path, 'rb') as f:
                    _, _, adj_mx = pickle.load(f)
            except UnicodeDecodeError as e:
                with open(self.adj_mx_path, 'rb') as f:
                    _, _, adj_mx = pickle.load(f, encoding='latin1')
            except Exception as e:
                print('Unable to load data ', self.adj_mx_path, ':', e)
                raise
            self.adj_mx = [self.asym_adj(adj_mx), self.asym_adj(np.transpose(adj_mx))]
            self.update_trainer_params["adj_mx"] = self.adj_mx
            self.update_model_params["adj_mx"] = self.adj_mx

        if self.distance_adj_mx_path is not None:
            self.distance_adj_mx = np.load(self.distance_adj_mx_path, allow_pickle=True)
            self.update_trainer_params["distance_adj_mx"] = self.distance_adj_mx
            self.update_model_params["distance_adj_mx"] = self.distance_adj_mx

        self.update_model_params["history_len"] = self.history_len
        self.update_model_params["out_dim"] = self.forecast_len
        self.update_dataset_params["history_len"] = self.history_len
        self.update_dataset_params["forecast_len"] = self.forecast_len
        self.update_trainer_params["forecast_len"] = self.forecast_len

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


# test
if __name__ == '__main__':
    data_path = 'data/regression_dataset/PEMS08/PEMS08.npz'
    adj_mx_path = 'data/regression_dataset/PEMS08/adj_PEMS08.pkl'
    distance_adj_mx_path = 'data/regression_dataset/PEMS08/adj_PEMS08_distance.pkl'
    data_preprocessor = PEMSDataPreprocessor(data_path, adj_mx_path=adj_mx_path, distance_adj_mx_path=distance_adj_mx_path)
    data = data_preprocessor.preprocess()
    print(data.shape)
    train_data, valid_data, test_data = data_preprocessor.split_data(data)
    print(train_data.shape, valid_data.shape, test_data.shape)
    print(data_preprocessor.update_trainer_params["adj_mx"].shape, data_preprocessor.update_trainer_params["distance_adj_mx"].shape)