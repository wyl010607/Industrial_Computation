import numpy as np
from .abs import AbstractDataPreprocessor
from utils.scaler import StandardScaler, MinMaxScaler
from utils.datasegment import get_loader_segment


class MCDataPreprocessor(AbstractDataPreprocessor):
    """
    Data Preprocessor for DCS data.
    """

    def __init__(
        self,
        data_path,
        win_size,
        step=1,
        dataset='SWAT',
        dist=0,
        ret_data=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.win_size = win_size
        self.step = step
        self.dataset = dataset
        self.dist = dist
        self.ret_data = ret_data
        self.update_dataset_params = {}
        self.update_model_params = {}
        self.update_trainer_params = {}
        # self.meta_batch = meta_batch
        # self.support_num = support_num
        # self.query_num = query_num


    def load_data(self):
        """
        Load data from the specified file path.

        This method loads the data, time stamps, and variable index dictionary
        from a file at `self.data_path`.
        """

    def preprocess(self):
        """
        Returns
        -------
        np.ndarray
            The preprocessed data array.
        """
        # Normalization
        #scaler = preprocessing.StandardScaler()

    def split_data(self):
        """
        Split the preprocessed data into training, validation, and testing sets.

        在时间序列异常检测任务中，需要分别按照训练集和测试集读取数据，在split_data模块需划分
        训练集与验证集的滑动窗口

        Parameters
        ----------
        preprocessed_data : np.ndarray
            The preprocessed data array to be split.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing the training, validation, and test data arrays, respectively.
        """
        train_dataset, train_sampler, train_shuffle = get_loader_segment(self.data_path, win_size=self.win_size,
                                               mode='train', dataset=self.dataset, step=self.step, dist=self.dist, ret_data = self.ret_data)
        val_dataset, val_sampler, val_shuffle = get_loader_segment(self.data_path, win_size=self.win_size,
                                              mode='val', dataset=self.dataset, step=self.step, dist=self.dist, ret_data = self.ret_data)
        test_dataset, test_sampler, test_shuffle = get_loader_segment(self.data_path, win_size=self.win_size,
                                              mode='test', dataset=self.dataset, step=self.step, dist=self.dist, ret_data = self.ret_data)
        thre_dataset, thre_sampler, thre_shuffle = get_loader_segment(self.data_path, win_size=self.win_size,
                                              mode='thre', dataset=self.dataset, step=self.step, dist=self.dist, ret_data = self.ret_data)

        return train_dataset, train_sampler, train_shuffle, val_dataset, val_sampler, val_shuffle, test_dataset, test_sampler, test_shuffle, thre_dataset, thre_sampler, thre_shuffle
