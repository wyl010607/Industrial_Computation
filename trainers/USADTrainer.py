from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import math
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from trainers.abs_usad import AbstractTrainer
from utils.early_stop import EarlyStopping

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class USADTrainer(AbstractTrainer):
    def __init__(
            self,
            model,
            scaler,
            scheduler,
            model_save_path,
            result_save_dir_path,
            max_epoch_num,
            res_pth,
            alpha=0.5,
            beta=0.5,
            enable_early_stop=False,
            early_stop_patience=3,
            early_stop_min_is_best=True,
            *args,
            **kwargs,
    ):
        super().__init__(
            model,
            scaler,
            scheduler,
            model_save_path,
            result_save_dir_path,
            max_epoch_num,
            enable_early_stop,
            early_stop_patience,
            early_stop_min_is_best,
            *args,
            **kwargs,
        )
        self.model = model
        self.optimizer1 = torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()))
        self.optimizer2 = torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()))
        self.alpha = alpha
        self.beta = beta

    def train_one_epoch(self, train_loader, val_loader, epoch, *args, **kwargs):
        total_loss1 = 0.0
        total_loss2 = 0.0
        path = self.model_save_path
        self.model.train()
        i = 0
        for [batch] in train_loader:
            batch = to_device(batch, self.device)

            # Train AE1
            loss1, loss2 = self.model.training_step(batch, epoch + 1)
            total_loss1 += loss1.item()
            loss1.backward()
            self.optimizer1.step()
            self.optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = self.model.training_step(batch, epoch + 1)
            total_loss2 += loss2.item()
            loss2.backward()
            self.optimizer2.step()
            self.optimizer2.zero_grad()

            i += 1

            if i % 200 == 0:
                avg_loss1 = total_loss1 / i
                avg_loss2 = total_loss2 / i
                print(f"Train loss1: {avg_loss1:.4f}, Train loss2: {avg_loss2:.4f}")
        print('epoch ({} / {}) (loss1:{:.8f}, loss2:{:.8f})'.format(
                epoch, self.max_epoch_num,
                total_loss1 / len(train_loader), total_loss2 / len(train_loader)), flush=True)
        return total_loss1,total_loss2

    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        self.model.eval()
        results = []
        with torch.no_grad():
            for [batch] in data_loader:
                batch = to_device(batch, self.device)
                w1 = self.model.decoder1(self.model.encoder(batch))
                w2 = self.model.decoder2(self.model.encoder(w1))
                results.append(
                    self.alpha * torch.mean((batch - w1) ** 2, axis=1) + self.beta * torch.mean((batch - w2) ** 2, axis=1))
        return results

