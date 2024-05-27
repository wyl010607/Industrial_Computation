import numpy as np


class EarlyStop:
    def __init__(self, patience, min_is_best):
        self.patience = patience
        self.min_is_best = min_is_best
        self.count = None
        finfo = np.finfo(np.float32)
        self.cur_values = finfo.max if self.min_is_best else finfo.min
        self.cur_values = None
        self.reset()

    def reset(self):
        self.count = 0
        finfo = np.finfo(np.float32)
        self.cur_value = finfo.max if self.min_is_best else finfo.min

    def reach_stop_criteria(self, cur_value):
        if self.min_is_best:
            self.count = self.count + 1 if cur_value >= self.cur_value else 0
        else:
            self.count = self.count + 1 if cur_value <= self.cur_value else 0
        if self.count == self.patience:
            return True
        if cur_value <= self.cur_value:
          self.cur_value = cur_value
        return False
