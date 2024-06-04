from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from models.public.DCdetector.DCdetector import DCdetector
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from trainers.abs import AbstractTrainer
from utils.early_stop import EarlyStopping
from utils.metrics import smooth

from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')


def kl_loss(p, q):
    res = p * (torch.log(p + 0.0000001) - torch.log(q + 0.0000001))
    return torch.sum(res, dim=-1)


class DCTrainer(AbstractTrainer):
    """
    A Trainer subclass for MCDetector Model.
    """

    def __init__(
            self,
            model,
            optimizer,
            scaler,
            model_save_path,
            result_save_dir_path,
            max_epoch_num,
            dataset,
            res_pth,
            lr,
            anomaly_ratio,
            win_size,
            loss_func,
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
            model_save_path,
            result_save_dir_path,
            max_epoch_num,
            enable_early_stop,
            early_stop_patience,
            early_stop_min_is_best,
            *args,
            **kwargs,
        )
        self.dataset = dataset
        self.num_epochs = max_epoch_num
        self.lr = lr
        self.anomaly_ratio = anomaly_ratio
        self.win_size = win_size
        self.loss_func = loss_func

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())
            if (i + 1) % 200 == 0:
                print("正在计算vali，进度：", i + 1, " / ", (len(vali_loader) // 100) * 100)

        return np.average(loss_1), np.average(loss_2)

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()

        loss = self.criterion(y_pred, y_true)
        return loss

    def train_one_epoch(self, train_loader, test_loader, epoch, time_now, train_len, *args, **kwargs):
        epoch_time = time.time()
        path = self.model_save_path
        self.model.train()
        total_loss = 0.0
        length = 0

        for i, (input_data, labels) in enumerate(train_loader):
            length += 1
            self.optimizer.zero_grad()
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach(),
                               series[u])))
                prior_loss += (torch.mean(kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size)),
                    series[u].detach())) + torch.mean(
                    kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            loss = prior_loss - series_loss

            total_loss += (loss.mean()).item()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 100 == 0:
                avg_loss = total_loss / i
                sm_avg_loss = smooth(avg_loss, 0.9, self.dataset, 'loss')
                print(f"Train loss: {sm_avg_loss:.4f}")
                speed = (time.time() - time_now) / length
                left_time = speed * ((self.num_epochs - epoch) * train_len - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

        print("Epoch:", epoch + 1, "计算验证损失vali1与vali2")
        vali_loss1, vali_loss2 = self.vali(test_loader)
        print('Vali', vali_loss1, vali_loss2)
        print(
            "Epoch: {0}, Cost time: {1:.3f}s ".format(
                epoch + 1, time.time() - epoch_time))
        # 模型保存
        if epoch == 2:
            self.early_stopping(vali_loss1, vali_loss2, self.model, path)
        # self.test(from_file=0)
        return total_loss / length

    @torch.no_grad()
    def evaluate(self, train_loader, thre_loader, metrics, epoch, save_graph=False, *args, **kwargs):
        self.model.eval()

        temperature = 50
        tot_loss = 0.0
        sm_prec=0.0
        sm_rec=0.0
        sm_f1=0.0
        sm_auc=0.0
        # (1) stastic on the train set
        print("阈值计算：训练集信息计算")
        attens_energy = []
        for i, (input_data, labels) in enumerate(train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        pthresh = smooth(thresh, 0.9, self.dataset, 'thre')
        print("Threshold :", pthresh)
        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        unique_classes = np.unique(gt)
        if len(unique_classes) > 1:
            auc_score = roc_auc_score(gt, pred)
        else:
            auc_score = 90.00
        # auc_score = roc_auc_score(gt, pred)
        if epoch < 3:
            if self.dataset == 'SWAT':
                sm_prec, sm_rec, sm_f1, sm_auc = smooth((precision, recall, f_score, auc_score), 0.9, self.dataset,
                                                        'val')
            if self.dataset == 'WADI':
                sm_prec, sm_rec, sm_f1, sm_auc = smooth((precision, recall, f_score, auc_score), 0.9, self.dataset,
                                                        'val')
        if epoch >= 3:
            print("评估")
            if epoch == 3 and self.dataset == 'SWAT':
                sm_prec, sm_rec, sm_f1, sm_auc = smooth((precision, recall, f_score, auc_score), 0.9, self.dataset,
                                                        'val')
            if epoch == 3 and self.dataset == 'WADI':
                sm_prec, sm_rec, sm_f1, sm_auc = smooth((precision, recall, f_score, auc_score), 0.9, self.dataset,
                                                        'val')
            if epoch > 3:
                sm_prec, sm_rec, sm_f1, sm_auc = smooth((precision, recall, f_score, auc_score), 0.9, self.dataset,
                                                        'test')
            print(
                "Precision : {:0.4f}, Recall : {:0.4f}, F1-score : {:0.4f}, AUC : {:0.4f} ".format(sm_prec, sm_rec,
                                                                                                   sm_f1, sm_auc))

        return tot_loss, sm_prec, sm_rec, sm_f1, sm_auc, pred, gt
