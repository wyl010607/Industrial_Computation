from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import math
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from trainers.abs import AbstractTrainer
from utils.early_stop import EarlyStopping
from utils.metrics import smooth

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSincePlus(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class GDNTrainer(AbstractTrainer):
    def __init__(
            self,
            model,
            optimizer,
            scaler,
            scheduler,
            model_save_path,
            result_save_dir_path,
            max_epoch_num,
            res_pth,
            enable_early_stop=False,
            early_stop_patience=3,
            early_stop_min_is_best=True,
            *args,
            **kwargs,
    ):
        super().__init__(
            model,
            optimizer,
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

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = F.mse_loss(y_pred, y_true, reduction='mean')

        return loss

    def train_one_epoch(self, train_loader, val_loader, epoch, *args, **kwargs):
        path = self.model_save_path
        self.model.train()
        i = 0

        acu_loss = 0
        train_loss_list = []
        for x, labels, attack_labels, edge_index in train_loader:
            _start = time.time()
            x, labels, edge_index = [item.float().to(self.device) for item in [x, labels, edge_index]]
            self.optimizer.zero_grad()
            out = self.model(x, edge_index).float().to(self.device)
            loss = self.loss_func(out, labels)

            loss.backward()
            self.optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
            i+=1
            if i % 200 == 0:
                avg_loss = acu_loss / i
                print(f"Train loss: {avg_loss:.4f}")
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
            epoch, self.max_epoch_num,
            acu_loss / len(train_loader), acu_loss), flush=True)
        return acu_loss

    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        # self.model.eval()
        test_loss_list = []
        now = time.time()
        test_predicted_list = []
        test_ground_list = []
        test_labels_list = []
        t_test_predicted_list = []
        t_test_ground_list = []
        t_test_labels_list = []
        test_len = len(data_loader)
        self.model.eval()

        i = 0
        acu_loss = 0
        for x, y, labels, edge_index in data_loader:
            x, y, labels, edge_index = [item.to(self.device).float() for item in [x, y, labels, edge_index]]
            with torch.no_grad():
                predicted = self.model(x, edge_index).float().to(self.device)
                loss = self.loss_func(predicted, y)
                labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])
                if len(t_test_predicted_list) <= 0:
                    t_test_predicted_list = predicted
                    t_test_ground_list = y
                    t_test_labels_list = labels
                else:
                    t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                    t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                    t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

            test_loss_list.append(loss.item())
            acu_loss += loss.item()
            i += 1
            if i % 1000 == 1 and i > 1:
                print(timeSincePlus(now, i / test_len))
        test_predicted_list = t_test_predicted_list.tolist()
        test_ground_list = t_test_ground_list.tolist()
        test_labels_list = t_test_labels_list.tolist()
        avg_loss = sum(test_loss_list) / len(test_loss_list)

        return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]
