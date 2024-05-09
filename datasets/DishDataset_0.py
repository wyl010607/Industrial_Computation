import numpy as np
import torch
from .abs import AbstractDataset


class DishDataset_0(AbstractDataset):

    def __init__(self, data, history_len, forecast_len, label_len,data_stamp, type,*args, **kwargs):

        super().__init__(data, *args, **kwargs)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        self.data = data
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.data_x= data
        self.data_y= data
        self.data_stamp=data_stamp

        self.label_len=label_len

        self.type = type

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.forecast_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if self.type=='train':
            data_stamp=self.data_stamp[0]
            seq_x_mark = data_stamp[s_begin:s_end]
            seq_y_mark = data_stamp[r_begin:r_end]
        elif self.type=='valid':
            data_stamp = self.data_stamp[1]
            seq_x_mark = data_stamp[s_begin:s_end]
            seq_y_mark = data_stamp[r_begin:r_end]
        else :
            data_stamp = self.data_stamp[2]
            seq_x_mark = data_stamp[s_begin:s_end]
            seq_y_mark = data_stamp[r_begin:r_end]


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.history_len - self.forecast_len + 1
