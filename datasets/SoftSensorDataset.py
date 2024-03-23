import numpy as np
import torch
from .abs import AbstractDataset


class SoftSensorDataset(AbstractDataset):
    """
    A dataset class for soft sensor.
    1 step in 1 step out
    """

    def __init__(self, data, input_index_list, output_index_list, *args, **kwargs):
        """
        Initialize the SoftSensorDataset.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            The dataset to be loaded. Should be a 2D array with dimensions (L, N)
                L: Sequence length
                N: Number of variables
                C: Number of channels
        input_index_list : list of int
            The index of the input variables
        output_index_list : list of int
            The index of the output variables
        """
        super().__init__(data, *args, **kwargs)
        assert len(data.shape) == 2
        self.data = data
        self.input_index_list = input_index_list
        self.output_index_list = output_index_list

    def __getitem__(self, index):
        """
        Retrieve a (X, y) pair from the dataset.

        Parameters
        ----------
        index : int
            The index

        Returns
        -------
        tuple of torch.Tensor

        """
        x = self.data[index, self.input_index_list]
        y = self.data[index, self.output_index_list]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __len__(self):
        return self.data.shape[0]
