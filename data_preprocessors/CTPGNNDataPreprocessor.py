import numpy as np
import torch

from .abs import AbstractDataPreprocessor
import pandas as pd

class CTPGNNDataPreprocessor(AbstractDataPreprocessor):
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
        self.adj_mx_path = kwargs.get("adj_mx_path")
        self.history_len = kwargs.get("history_len")
        self.forecast_len = kwargs.get("forecast_len")
        self.eps = kwargs.get("eps")

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

    def weight_matrix_nl(self, file_path, sigma2=0.1, epsilon=0.1, scaling=True):
        '''
        Load weight matrix function.
        :param file_path: str, the path of saved weight matrix file.
        :param sigma2: float, scalar of matrix W.
        :param epsilon: float, thresholds to control the sparsity of matrix W.
        :param scaling: bool, whether applies numerical scaling on W.
        :return: np.ndarray, [n_route, n_route].
        '''
        try:
            W = np.load((file_path))
        except FileNotFoundError:
            print(f'ERROR: input file was not found in {file_path}.')

        # check whether W is a 0/1 matrix.
        if set(np.unique(W)) == {0, 1}:
            print('The input graph is a 0/1 matrix; set "scaling" to False.')
            scaling = False

        if scaling:
            n = W.shape[0]
            W = W / 10000.
            W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
            # refer to Eq.10
            W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
            # return laplacian(W)
            # print((W>0).sum()/(W.shape[0])**2)
            return W
        else:
            return W
    def load_data(self):
        """
        Load data from the specified file path.

        This method loads the data, time stamps, and variable index dictionary
        from a file at `self.data_path`.
        """
        data = np.load(self.data_path, allow_pickle=True)
        self.data, self.time_stamp_array, self.vars_index_dict = (
            data["data_array"],
            data["time_stamp_array"],
            data["vars_index_dict"].tolist(),
        )


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
        if self.adj_mx_path is not None:
            self.adj_mx = self.weight_matrix_nl(self.adj_mx_path, epsilon=self.eps)
            self.adj_mx = torch.from_numpy(self.adj_mx).float().cuda()
            self.update_trainer_params["adj_mx"] = self.adj_mx
        self.update_trainer_params["forecast_len"] = self.forecast_len
        self.update_trainer_params["history_len"] = self.history_len
        self.update_trainer_params["num_route"] = self.data.shape[1]

        self.update_model_params["adj_mx"] = self.adj_mx
        self.update_model_params["history_len"] = self.history_len
        self.update_model_params["forecast_len"] = self.forecast_len
        self.update_model_params["num_route"] = self.data.shape[1]
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
