import numpy as np
import torch
import pandas as pd
from scipy.optimize import linprog
from .abs import AbstractDataPreprocessor


class PEMSD7DataPreprocessor(AbstractDataPreprocessor):

    def __init__(
        self,
        data_path,
        *args,
        **kwargs
    ):
        super().__init__(data_path, *args, **kwargs)
        self.data = None
        self.adj_mx = None
        self.adj_mx_path = kwargs.get("adj_mx_path")
        self.stamp_path = kwargs.get("stamp_path")
        self.history_len = kwargs.get("history_len")
        self.forecast_len = kwargs.get("forecast_len")
        self.eps = kwargs.get("eps")
        self.load_data()

    def normalize(self, a):
        mu = np.mean(a, axis=1, keepdims=True)
        std = np.std(a, axis=1, keepdims=True)
        return (a - mu) / std

    def wasserstein_distance(self, p, q, D):
        A_eq = []
        for i in range(len(p)):
            A = np.zeros_like(D)
            A[i, :] = 1
            A_eq.append(A.reshape(-1))
        for i in range(len(q)):
            A = np.zeros_like(D)
            A[:, i] = 1
            A_eq.append(A.reshape(-1))
        A_eq = np.array(A_eq)
        b_eq = np.concatenate([p, q])
        D = np.array(D)
        D = D.reshape(-1)

        result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
        myresult = result.fun

        return myresult

    def spatial_temporal_aware_distance(self, x, y):
        x, y = np.array(x), np.array(y)
        x_norm = (x ** 2).sum(axis=1, keepdims=True) ** 0.5
        y_norm = (y ** 2).sum(axis=1, keepdims=True) ** 0.5
        p = x_norm[:, 0] / x_norm.sum()
        q = y_norm[:, 0] / y_norm.sum()
        D = 1 - np.dot(x / x_norm, (y / y_norm).T)
        return self.wasserstein_distance(p, q, D)

    def spatial_temporal_similarity(self, x, y, normal, transpose):
        if normal:
            x = self.normalize(x)
            x = self.normalize(x)
        if transpose:
            x = np.transpose(x)
            y = np.transpose(y)
        return 1 - self.spatial_temporal_aware_distance(x, y)

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
            W = pd.read_csv(file_path, header=None).values
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
        self.data = pd.read_csv(self.data_path, header=None).values.astype(float)  # -> np.ndarray



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
        T, N = self.data.shape
        data1 = self.data.reshape(-1, 288, N)
        d = np.zeros([N, N])
        for i in range(N):
            for j in range(i + 1, N):
                d[i, j] = self.spatial_temporal_similarity(data1[:, :, i], data1[:, :, j], normal=False, transpose=False)

        adj = d+d.T
        np.save("data/stag_001_.npy", adj)
        adj = np.load("data/stag_001_.npy")
        id_mat = np.identity(N)
        adjl = adj + id_mat
        adjlnormd = adjl / adjl.mean(axis=0)

        adj = 1 - adjl + id_mat
        A_adj = np.zeros([N, N])
        R_adj = np.zeros([N, N])
        # A_adj = adj
        adj_percent = 0.01

        top = int(N * adj_percent)

        for i in range(adj.shape[0]):
            a = adj[i, :].argsort()[0:top]
            for j in range(top):
                A_adj[i, a[j]] = 1
                R_adj[i, a[j]] = adjlnormd[i, a[j]]

        for i in range(N):
            for j in range(N):
                if (i == j):
                    R_adj[i][j] = adjlnormd[i, j]
        print("The weighted matrix of temporal graph is generated!")

        self.update_dataset_params["history_len"] = self.history_len
        self.update_dataset_params["forecast_len"] = self.forecast_len
        self.update_dataset_params["stamp_path"] = self.stamp_path

        # load adjacency matrix
        if self.adj_mx_path is not None:
            self.adj_mx = self.weight_matrix_nl(self.adj_mx_path, epsilon=self.eps)
            self.adj_mx = torch.from_numpy(self.adj_mx).float().cuda()
            self.update_trainer_params["adj_mx"] = self.adj_mx
        self.update_trainer_params["forecast_len"] = self.forecast_len
        self.update_trainer_params["history_len"] = self.history_len
        self.update_trainer_params["num_route"] = self.data.shape[1]

        self.update_model_params["adj_mx"] = self.adj_mx
        self.update_model_params["sta"] = R_adj
        self.update_model_params["history_len"] = self.history_len
        self.update_model_params["forecast_len"] = self.forecast_len
        self.update_model_params["num_route"] = self.data.shape[1]
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
        total_length = len(preprocessed_data)
        train_end = int(self.train_ratio * total_length+0.5)
        valid_end = int((self.train_ratio + self.valid_ratio) * total_length+0.5)

        train_data = preprocessed_data[:train_end]
        valid_data = preprocessed_data[train_end:valid_end]
        test_data = preprocessed_data[valid_end:]

        return train_data, valid_data, test_data

