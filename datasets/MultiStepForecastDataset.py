import numpy as np
import torch
from .abs import AbstractDataset


class MultiStepForecastDataset(AbstractDataset):
    """
    A dataset class for multi-step time series forecasting.
    Single step forecasting is a special case of multi-step forecasting when set forecast_len=1.
    """

    def __init__(self, data, history_len, forecast_len, long_history_len=None, *args, **kwargs):
        """
        Initialize the MultiStepForecastDataset.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            The dataset to be loaded. Should be a 2D array with dimensions (L, N)(which will be reshaped)
            or a 3D array with dimensions (L, N, C), where:
                L: Sequence length
                N: Number of variables
                C: Number of channels
        history_len : int
            The length of the historical data sequence.
        forecast_len : int
            The length of the forecast data sequence.
        long_history_len : int, optional
            The length of the long historical data sequence for pretrain (default is None).
        """
        super().__init__(data, *args, **kwargs)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        self.data = data
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.long_history_len = long_history_len

    def __getitem__(self, index):
        """
        Retrieve a (history, forecast) pair from the dataset.

        Parameters
        ----------
        index : int
            The index of the historical sequence to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing two tensors: the historical data (x) with dimensions (history_len, N, C) and the forecast
            data (y) with dimensions (forecast_len, N, C), both as torch.float32 tensors.
        """
        if self.long_history_len is not None:
            long_history_data = self.data[index: index + self.long_history_len]
            history_data = self.data[
                           index
                           + self.long_history_len
                           - self.history_len: index
                                               + self.long_history_len
                           ]
            future_data = self.data[
                          index
                          + self.long_history_len: index
                                                   + self.long_history_len
                                                   + self.forecast_len
                          ]

            return (
                torch.from_numpy(history_data).type(torch.float32),
                torch.from_numpy(future_data).type(torch.float32),
                torch.from_numpy(long_history_data).type(torch.float32),
            )
        else:
            x = self.data[index : index + self.history_len]
            y = self.data[
                index + self.history_len : index + self.history_len + self.forecast_len
            ]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32
            )

    def __len__(self):
        if self.long_history_len is not None:
            return self.data.shape[0] - self.long_history_len - self.forecast_len + 1
        else:
            return self.data.shape[0] - self.history_len - self.forecast_len + 1
