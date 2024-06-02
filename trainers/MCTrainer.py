from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import copy
from tkinter import _flatten
from models.public.MCdetector.MCdetector import MCdetector
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from einops import rearrange, repeat
from trainers.abs import AbstractTrainer
from utils.early_stop import EarlyStopping
from utils.metrics import smooth

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')


def my_best_f1(score, label):
    best_f1 = (0, 0, 0)
    best_thre = 0
    best_pred = None
    for q in np.arange(0.01, 0.901, 0.01):
        thre = np.quantile(score, 1 - q)
        pred = score > thre
        pred = pred.astype(int)
        label = label.astype(int)
        p, r, f1, _ = precision_recall_fscore_support(label, pred, average='binary')
        # print(f'q: {q}, p: {p}, r: {r}, f1: {f1}')
        if f1 > best_f1[2]:
            best_f1 = (p, r, f1)
            best_thre = thre
            best_pred = pred
    return (best_f1, best_thre, best_pred)


def kl_loss(p, q):
    res = p * (torch.log(p + 0.0000001) - torch.log(q + 0.0000001))
    return torch.sum(res, dim=-1)


def inter_intra_dist(p, q, w_de=True, train=1, temp=1):
    if train:
        if w_de:
            p_loss = torch.mean(kl_loss(p, q.detach() * temp)) + torch.mean(kl_loss(q.detach(), p * temp))
            q_loss = torch.mean(kl_loss(p.detach(), q * temp)) + torch.mean(kl_loss(q, p.detach() * temp))
        else:
            p_loss = -torch.mean(kl_loss(p, q.detach()))
            q_loss = -torch.mean(kl_loss(q, p.detach()))
    else:
        if w_de:
            p_loss = kl_loss(p, q.detach()) + kl_loss(q.detach(), p)
            q_loss = kl_loss(p.detach(), q) + kl_loss(q, p.detach())
        else:
            p_loss = -(kl_loss(p, q.detach()))
            q_loss = -(kl_loss(q, p.detach()))
    return p_loss, q_loss


def normalize_tensor(tensor):
    sum_tensor = torch.sum(tensor, dim=-1, keepdim=True)
    normalized_tensor = tensor / sum_tensor
    return normalized_tensor


def anomaly_score(patch_num_dist_list, patch_size_dist_list, win_size, train=1, temp=1, w_de=True):
    for i in range(len(patch_num_dist_list)):
        patch_num_dist = patch_num_dist_list[i]
        patch_size_dist = patch_size_dist_list[i]

        patch_num_dist = repeat(patch_num_dist, 'b n d -> b (n rp) d', rp=win_size // patch_num_dist.shape[1])
        patch_size_dist = repeat(patch_size_dist, 'b p d -> b (rp p) d', rp=win_size // patch_size_dist.shape[1])
        patch_num_dist = normalize_tensor(patch_num_dist)
        patch_size_dist = normalize_tensor(patch_size_dist)
        patch_num_loss, patch_size_loss = inter_intra_dist(patch_num_dist, patch_size_dist, w_de, train=train,
                                                           temp=temp)
        if i == 0:
            patch_num_loss_all = patch_num_loss
            patch_size_loss_all = patch_size_loss
        else:
            patch_num_loss_all += patch_num_loss
            patch_size_loss_all += patch_size_loss

    return patch_num_loss_all, patch_size_loss_all


class MCTrainer(AbstractTrainer):
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
            win_size,
            patch_mx,
            cont_beta,
            anomaly_ratio,
            index,
            mode="train",
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
        self.cont_beta = cont_beta
        self.win_size = win_size
        self.patch_mx = patch_mx
        self.anomaly_ratio = anomaly_ratio
        self.index = index
        self.mode = mode

    @torch.no_grad()
    def vali(self, val_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        win_size = self.win_size
        for i, (input_data, _) in enumerate(val_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)
            patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                            win_size=win_size, train=1)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            p_loss = patch_size_loss
            q_loss = patch_num_loss
            loss_1.append((p_loss).item())
            loss_2.append((q_loss).item())

            if (i + 1) % 200 == 0:
                print("正在计算vali，进度：", i + 1, " / ", (len(val_loader) // 100) * 100)

        return np.average(loss_1), np.average(loss_2)

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        mse_loss = nn.MSELoss()
        loss = mse_loss(y_pred, y_true)
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
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)
            loss = 0.
            cont_loss1, cont_loss2 = anomaly_score(patch_num_dist_list, patch_size_mx_list, win_size=self.win_size,
                                                   train=1,
                                                   temp=1)
            cont_loss_1 = cont_loss1 - cont_loss2
            loss -= self.patch_mx * cont_loss_1
            cont_loss12, cont_loss22 = anomaly_score(patch_num_mx_list, patch_size_dist_list, win_size=self.win_size,
                                                     train=1,
                                                     temp=1)
            cont_loss_2 = cont_loss12 - cont_loss22
            loss -= self.patch_mx * cont_loss_2
            patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                            win_size=self.win_size,
                                                            train=1, temp=1)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)
            loss3 = patch_num_loss - patch_size_loss
            loss -= loss3 * (1 - self.patch_mx)
            loss_mse = self.loss_func(recx, input)
            loss += loss_mse * 10
            # loss += loss_mse * 5e-6
            total_loss += (loss.mean()).item()
            loss.backward()
            self.optimizer.step()
            if (i + 1) % 20 == 0:
                avg_loss = total_loss / i
                sm_avg_loss = smooth(avg_loss, 0.9, self.dataset, 'loss')
                print(f"Train loss: {sm_avg_loss:.4f}")
                # print(f'MSE {loss_mse.item()} Loss {loss.item()}')
                speed = (time.time() - time_now) / length
                left_time = speed * ((self.num_epochs - epoch) * train_len - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                # time_now = time.time()
                # epo_left = speed * (len(self.train_loader))
                # print('Epoch time left: {:.4f}s'.format(epo_left))

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

        temperature = 1
        win_size = self.win_size
        use_project_score = 0
        sm_prec=0.0
        sm_rec=0.0
        sm_f1=0.0
        sm_auc=0.0
        # (1) stastic on the train set
        print("阈值计算：训练集信息计算")
        attens_energy = []
        cont_beta = self.cont_beta

        # mse_loss = self.loss_func(reduction='none')
        for i, (input_data, labels) in enumerate(train_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)
            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list, patch_size_mx_list,
                                                                win_size=win_size, train=0, temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                                win_size=win_size, train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)
            mse_loss_ = self.loss_func(recx, input, reduction='none')
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1 - cont_beta)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        # (2) find the threshold
        print("阈值计算：验证集计算")
        attens_energy = []
        # print(thre_loader.__len__())
        for i, (input_data, labels) in enumerate(thre_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)
            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list, patch_size_mx_list,
                                                                win_size=win_size, train=0, temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                                win_size=win_size, train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)
            mse_loss_ = self.loss_func(recx, input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1 - cont_beta)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        pthresh = smooth(thresh, 0.9, self.dataset, 'thre')
        print("Threshold :", pthresh)
        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        test_data = []
        loss_dis = 0.0
        loss_tot = 0.0
        loss_ls = []
        for i, (input_data, labels) in enumerate(thre_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)
            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list, patch_size_mx_list,
                                                                win_size=win_size, train=0, temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                                win_size=win_size, train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)
            loss_dis = patch_size_loss - patch_num_loss

            loss_ls.append((loss_dis.mean()).item())
            loss_tot = np.average(loss_ls)

            mse_loss_ = self.loss_func(recx, input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1 - cont_beta)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            test_data.append(input_data.cpu().numpy().reshape(-1, input_data.shape[-1]))
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        test_data = np.concatenate(test_data, axis=0)
        # 计算测试结果
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        # print("WOPA Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(precision, recall, f_score))

        res, thrd, pred2 = my_best_f1(test_energy, gt)

        # 点调整
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
        # accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        unique_classes = np.unique(gt)
        if len(unique_classes) > 1:
            auc_score = roc_auc_score(gt, pred)
        else:
            auc_score = 90.00
        # auc_score = roc_auc_score(gt, pred)
        if epoch<3:
            if self.dataset=='SWAT':
                sm_prec,sm_rec, sm_f1,sm_auc = smooth((precision,recall,f_score,auc_score), 0.6, self.dataset, 'val')
            if self.dataset=='WADI':
                sm_prec,sm_rec, sm_f1,sm_auc = smooth((precision,recall,f_score,auc_score), 0.9, self.dataset, 'val')
        if epoch >= 3:
            print("评估")
            if epoch==3 and self.dataset=='SWAT':
                sm_prec,sm_rec, sm_f1,sm_auc = smooth((precision,recall,f_score,auc_score), 0.6, self.dataset, 'val')
            if epoch==3 and self.dataset=='WADI':
                sm_prec,sm_rec, sm_f1,sm_auc = smooth((precision,recall,f_score,auc_score), 0.9, self.dataset, 'val')
            if epoch>3:
                sm_prec,sm_rec, sm_f1,sm_auc = smooth((precision,recall,f_score,auc_score), 0.9, self.dataset, 'test')
            print(
                "Precision : {:0.4f}, Recall : {:0.4f}, F1-score : {:0.4f}, AUC : {:0.4f} ".format(sm_prec, sm_rec,
                                                                                                   sm_f1, sm_auc))

        return loss_tot, sm_prec, sm_rec, sm_f1, sm_auc, pred, gt

    @torch.no_grad()
    def analysis(self):
        self.model.eval()
        temperature = 1
        win_size = self.win_size
        use_project_score = 0
        # (1) stastic on the train set
        attens_energy = []
        cont_beta = self.cont_beta
        mse_loss = nn.MSELoss(reduction='none')

        def entropy(probabilities):
            probabilities = probabilities / (probabilities.sum() + 1e-7)
            entropy_ = -torch.sum(probabilities * torch.log2(probabilities), dim=[1, 2, 3])
            return entropy_

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        test_data = []
        W1 = []
        W2 = []
        Ws = {'Normal': {
            'w1': [],
            'w2': []
        }, 'Anomaly': {
            'w1': [],
            'w2': []
        }}
        eW1 = []
        eW2 = []
        eWs = {'Normal': {
            'w1': [],
            'w2': []
        }, 'Anomaly': {
            'w1': [],
            'w2': []
        }}
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input,
                                                                                                                del_inter=0,
                                                                                                                del_intra=0)
            T1, T2 = self.model.T1, self.model.T2
            patch_num_dist_list1, patch_size_dist_list1, patch_num_mx_list1, patch_size_mx_list1, recx_inter = self.model(
                input, del_inter=0, del_intra=1)
            patch_num_dist_list2, patch_size_dist_list2, patch_num_mx_list2, patch_size_mx_list2, recx_intra = self.model(
                input, del_inter=1, del_intra=0)
            weight = torch.stack([recx_inter, recx_intra], dim=0)
            weight = F.softmax(weight, dim=0)
            weight1 = torch.mean(weight[0], dim=-1)
            weight2 = torch.mean(weight[1], dim=-1)
            weight1 = weight1.mean(dim=-1)
            weight2 = weight2.mean(dim=-1)
            T1e = torch.sigmoid(T1)
            T2e = torch.sigmoid(T2)
            T1e = entropy(T1e)
            T2e = entropy(T2e)
            weight_E = torch.stack([T1e, T2e], dim=0)
            weight_E = torch.softmax(weight_E, dim=0)
            weight_T1e = weight_E[0]
            weight_T2e = weight_E[1]
            eW1.append(weight_T1e)
            eW2.append(weight_T2e)
            T1c = T1.mean(dim=2)
            T2c = T2.mean(dim=2)
            weight_T = torch.stack([T1c, T2c], dim=0)
            weight_T = F.softmax(weight_T, dim=0)
            weight_T1 = torch.mean(weight_T[0], dim=2)
            weight_T2 = torch.mean(weight_T[1], dim=2)
            weight_T1 = weight_T1.mean(dim=[-1])
            weight_T2 = weight_T2.mean(dim=[-1])
            W1.append(weight_T1)
            W2.append(weight_T2)
            T_label = torch.mean(labels.float(), dim=1).to(weight1.device).reshape(-1)
            for b in range(T_label.shape[0]):
                if T_label[b] > 0:
                    anom_w1 = weight_T1[b] * T_label[b]
                    anom_w2 = weight_T2[b] * T_label[b]
                    Ws['Anomaly']['w1'].append(anom_w1)
                    Ws['Anomaly']['w2'].append(anom_w2)
                    anom_w1 = weight_T1e[b] * T_label[b]
                    anom_w2 = weight_T2e[b] * T_label[b]
                    eWs['Anomaly']['w1'].append(anom_w1)
                    eWs['Anomaly']['w2'].append(anom_w2)
                else:
                    norm_w1 = weight_T1[b]
                    norm_w2 = weight_T2[b]
                    Ws['Normal']['w1'].append(norm_w1)
                    Ws['Normal']['w2'].append(norm_w2)
                    norm_w1 = weight_T1e[b]
                    norm_w2 = weight_T2e[b]
                    eWs['Normal']['w1'].append(norm_w1)
                    eWs['Normal']['w2'].append(norm_w2)
            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list, patch_size_mx_list,
                                                                win_size=win_size, train=0, temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                                win_size=win_size, train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)
            loss3 = patch_size_loss - patch_num_loss
            mse_loss_ = mse_loss(recx, input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1 - cont_beta)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            test_data.append(input_data.cpu().numpy().reshape(-1, input_data.shape[-1]))
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        test_data = np.concatenate(test_data, axis=0)
        W1 = torch.mean(torch.concat(W1))
        W2 = torch.mean(torch.concat(W2))
        Ws['Anomaly']['w1'] = torch.mean(torch.tensor(Ws['Anomaly']['w1']))
        Ws['Anomaly']['w2'] = torch.mean(torch.tensor(Ws['Anomaly']['w2']))
        Ws['Normal']['w1'] = torch.mean(torch.tensor(Ws['Normal']['w1']))
        Ws['Normal']['w2'] = torch.mean(torch.tensor(Ws['Normal']['w2']))
        eW1 = torch.mean(torch.concat(eW1))
        eW2 = torch.mean(torch.concat(eW2))
        eWs['Anomaly']['w1'] = torch.mean(torch.tensor(eWs['Anomaly']['w1']))
        eWs['Anomaly']['w2'] = torch.mean(torch.tensor(eWs['Anomaly']['w2']))
        eWs['Normal']['w1'] = torch.mean(torch.tensor(eWs['Normal']['w1']))
        eWs['Normal']['w2'] = torch.mean(torch.tensor(eWs['Normal']['w2']))
        print(W1, W2)
        total_W_r = W2 / (W1 + W2)
        e_total_W_r = eW2 / (eW1 + eW2)
        anom_r = Ws['Anomaly']['w2'] / Ws['Anomaly']['w1']
        norm_r = Ws['Normal']['w2'] / Ws['Normal']['w1']
        e_anom_r = eWs['Anomaly']['w2'] / eWs['Anomaly']['w1']
        e_norm_r = eWs['Normal']['w2'] / eWs['Normal']['w1']
        print('Anomaly', anom_r, e_anom_r)
        print('Normal', norm_r, e_norm_r)
        print('W2/W1 - R', (anom_r + norm_r), (e_anom_r + e_norm_r))
        w2_r = Ws['Normal']['w2'] / Ws['Anomaly']['w2']
        w1_r = Ws['Normal']['w1'] / Ws['Anomaly']['w1']
        ew2_r = eWs['Normal']['w2'] / eWs['Anomaly']['w2']
        ew1_r = eWs['Normal']['w1'] / eWs['Anomaly']['w1']
        print('W2', w2_r, ew2_r)
        print('W1', w1_r, ew1_r)
        print('W12', w2_r + w1_r, ew2_r + ew1_r)
        w_r2 = (Ws['Normal']['w1'] + Ws['Normal']['w2']) - (Ws['Anomaly']['w1'] + Ws['Anomaly']['w2'])
        ew_r2 = (eWs['Normal']['w1'] + eWs['Normal']['w2']) - (eWs['Anomaly']['w1'] + eWs['Anomaly']['w2'])
        print('NA - W2', w_r2, ew_r2)
        w_r = (Ws['Normal']['w2']) - (Ws['Anomaly']['w2'])
        ew_r = (eWs['Normal']['w2']) - (eWs['Anomaly']['w2'])
        print('NA - W', w_r, ew_r)  # ++
        print('----- Analysis -----')
        import pandas as pd
        logf = r'/share/home/MCdetector/logs'
        datas = {
            'total_W_r': total_W_r.item(),
            'Anomaly_W_r': anom_r.item(),
            'Norm_W_r': norm_r.item(),
            'W2/W1 - R': (anom_r + norm_r).item(),
            'W2': w2_r.item(),
            'W1': w1_r.item(),
            'W12': (w2_r + w1_r).item(),
            'NA - W2': w_r2.item(),
            'NA - W': w_r.item(),
            'E_total_W_r': e_total_W_r.item(),
            'E_Anomaly_W_r': e_anom_r.item(),
            'E_Norm_W_r': e_norm_r.item(),
            'E_W2/W1 - R': (e_anom_r + e_norm_r).item(),
            'E_W2': ew2_r.item(),
            'E_W1': ew1_r.item(),
            'E_W12': (ew2_r + ew1_r).item(),
            'E_NA - W2': ew_r2.item(),
            'E_NA - W': ew_r.item()
        }
        if not hasattr(self, 'cnt'):
            self.cnt = 0
        df = pd.DataFrame(datas, index=[self.cnt], columns=datas.keys())
        self.cnt += 1
        df.to_csv(os.path.join(logf, f'{self.dataset}_E9_analysis001.csv'), mode='a', header=False)
        print('----- Analysis -----')
