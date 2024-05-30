import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import math


class TaskDataset(Dataset):
    def __init__(self, data, meta_batch, support_num, query_num, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.data = data
        self.meta_batch = meta_batch
        self.support_num = support_num
        self.query_num = query_num
        self.task_num = self.meta_batch // (self.support_num+self.query_num)
        self.task_sample_num = self.support_num+self.query_num
        self.window = support_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def task_split(self, data, support, query, task_num, meta_batch):
        """
        从数据集中采样一个任务，包含支持集和查询集。

        参数:
        - data: 包含输入训练数据集
        返回:
        - support_data: 采样的得到的包含所有元批次训练数据的支持集
        - query_data： 采样的得到的包含所有元批次验证数据的查询集
        """
        support_data = []
        query_data = []
        batch_num = math.ceil(len(data)/meta_batch)
        for k in range(batch_num):
            support_batch = []
            query_batch = []
            for i in range(task_num):
                support_raw = []
                query_raw = []
                meta_samples = random.sample(data, support+query)
                support_raw.extend(meta_samples[:support])
                query_raw.extend(meta_samples[support:])
                support_batch.append(support_raw)
                query_batch.append(query_raw)
            support_data.append(support_batch)
            query_data.append(query_batch)
        return support_data, query_data
