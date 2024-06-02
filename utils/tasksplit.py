import math
import random
import numpy as np
from utils.func import sample_ch


class TaskDataprocess:
    def __init__(self, data, meta_batch, support_num, query_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.meta_batch = meta_batch
        self.support_num = support_num
        self.query_num = query_num
        self.task_num = self.meta_batch // (self.support_num+self.query_num)
        self.task_sample_num = self.support_num+self.query_num
        self.window = support_num

    def task_split(self):
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
        batch_num = math.ceil(len(self.data)/self.meta_batch)
        judge = sample_ch(self.meta_batch)
        print("===开始对MAML模块中的支持集与查询集数据采样===")
        if judge==1:
            for k in range(batch_num):
                support_batch = []
                query_batch = []
                for i in range(self.task_num):
                    support_raw = []
                    query_raw = []
                    meta_samples = np.random.choice(self.data, 1, replace=False)
                    support_raw.extend(meta_samples[:self.support_num])
                    query_raw.extend(meta_samples[self.support_num:])
                    support_batch.append(support_raw)
                    query_batch.append(query_raw)
        return support_data, query_data
