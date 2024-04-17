import numpy as np
import torch
from .abs import AbstractDataset


class COTPGNETDataset(AbstractDataset):
    """
    A dataset class for multi-step time series forecasting.
    Single step forecasting is a special case of multi-step forecasting when set forecast_len=1.
    """

    def __init__(self, data, history_len, forecast_len, type, *args, **kwargs):
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
        """
        super().__init__(data, *args, **kwargs)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        self.data = data
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.type = type

        T, N, C = data.shape
        time_stamp = np.zeros(T)
        for idx in range(T):
            time_stamp[idx] = idx % 228

        self.time_stamp = time_stamp

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

        s_begin = index
        s_end = s_begin + self.history_len
        r_begin = s_end
        r_end = r_begin + self.forecast_len

        x = self.data[s_begin:s_end]
        y = self.data[r_begin:r_end]
        stamp_x = self.time_stamp[s_begin:s_end]
        stamp_y = self.time_stamp[r_begin:r_end]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(stamp_x, dtype=torch.float32), torch.tensor(stamp_y, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0] - self.history_len - self.forecast_len + 1
