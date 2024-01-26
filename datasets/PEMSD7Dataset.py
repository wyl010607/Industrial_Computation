import numpy as np
import torch
from .abs import AbstractDataset


class PEMSD7Dataset(AbstractDataset):

    def __init__(self, data, type, day_slot, num_route, history_len, forecast_len, stamp_path, T4N_step, *args, **kwargs):
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
        self.data = torch.Tensor(data)
        time_stamp = torch.Tensor(np.load(stamp_path))

        num_day = len(data) // day_slot
        num_slot = 288 * num_day

        if type == "train":
            num_slot = day_slot - history_len - forecast_len - T4N_step + 2
            x = torch.zeros(num_day * num_slot, 1, history_len, num_route)
            stamp = torch.zeros(num_day * num_slot, history_len)
            y = torch.zeros(num_day * num_slot, 1, forecast_len + T4N_step - 1, num_route)

        else:
            num_slot = day_slot - history_len - forecast_len + 1
            x = torch.zeros(num_day * num_slot, 1, history_len, num_route)
            stamp = torch.zeros(num_day * num_slot, history_len)
            y = torch.zeros(num_day * num_slot, 1, forecast_len, num_route)

        for i in range(num_day):
            for j in range(num_slot):
                t = i * num_slot + j
                s = i * day_slot + j
                e = s + history_len
                x[t, :, :, :] = self.data[s: e].reshape(1, history_len, num_route)
                stamp[t, :] = time_stamp[s:e].reshape(history_len)
                if type == "train":
                    length = forecast_len + T4N_step - 1
                    y[t, :, :, :] = self.data[e: e + length].reshape(1, length, num_route)
                else:
                    y[t, :, :, :] = self.data[e: e + forecast_len].reshape(1, forecast_len, num_route)
        self.stamp = stamp
        self.x = x.permute(0, 2, 3, 1)  # [slots, 1, history_len, num_route] -> [slots, history_len, num_route,  1]
        self.y = y.permute(0, 2, 3, 1)  # [slots, 1, forcast_len, num_route] -> [slots, forcast_len, num_route,  1]

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
        return self.x[index], self.stamp[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]  # self.dataset.shape = slots * routes = [slots, 228]
