from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score
from einops import rearrange,repeat
from trainers.abs import AbstractTrainer


class MCTrainer(AbstractTrainer):
    """
    A Trainer subclass for MCDetector Model.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        model_save_path,
        result_save_dir_path,
        max_epoch_num,
        enable_early_stop=False,
        early_stop_patience=3,
        early_stop_min_is_best=True,
        window=100,
        thresh=0.8,
        hyper_c=0.5,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            The neural network model for single step forecasting.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training the model.
        scheduler : torch.optim.lr_scheduler
            The learning rate scheduler.
        scaler : Scaler
            Scaler object used for normalizing and denormalizing data.
        model_save_path : str
            Path to save the trained model.
        result_save_dir_path : str
            Directory path to save training and evaluation results.
        max_epoch_num : int
            The maximum number of epochs for training.
        enable_early_stop : bool, optional
            Whether to enable early stopping (default is False).
        early_stop_patience : int, optional
            Number of epochs with no improvement after which training will be stopped (default is 5).
        early_stop_min_is_best : bool, optional
            Flag to indicate if lower values of loss indicate better performance (default is True).
        """
        super().__init__(
            model,
            optimizer,
            scheduler,
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
        self.update_dataset_params = {}
        self.update_model_params = {}
        self.update_trainer_params = {}
        self.window = window
        self.thresh = thresh
        self.hyper_c = hyper_c

    def best_f1(self, score, label):
        best_f1 = (0, 0, 0)
        best_thre = 0
        best_pred = None
        for q in np.arange(0.01, 0.901, 0.01):
            thre = np.quantile(score, 1 - q)
            pred = score > thre
            pred = pred.astype(int)
            label = label.astype(int)
            p, r, f1, _ = precision_recall_fscore_support(label, pred, average='binary')
            print(f'q: {q}, p: {p}, r: {r}, f1: {f1}')
            if f1 > best_f1[2]:
                best_f1 = (p, r, f1)
                best_thre = thre
                best_pred = pred

        return (best_f1, best_thre, best_pred)

    def kl_loss(self, p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.SmoothL1Loss()(y_pred, y_true)
        return loss.mean()

    def rep_loss(self, p, q, w_de=True, train=1, temp=1):
        if train:
            if w_de:
                p_loss = torch.mean(self.kl_loss(p, q.detach() * temp)) + torch.mean(self.kl_loss(q.detach(), p * temp))
                q_loss = torch.mean(self.kl_loss(p.detach(), q * temp)) + torch.mean(self.kl_loss(q, p.detach() * temp))
            else:
                p_loss = -torch.mean(self.kl_loss(p, q.detach()))
                q_loss = -torch.mean(self.kl_loss(q, p.detach()))
        else:
            if w_de:
                p_loss = self.kl_loss(p, q.detach()) + self.kl_loss(q.detach(), p)
                q_loss = self.kl_loss(p.detach(), q) + self.kl_loss(q, p.detach())

            else:
                p_loss = -(self.kl_loss(p, q.detach()))
                q_loss = -(self.kl_loss(q, p.detach()))
        return p_loss, q_loss

    def anomaly_score(self, patch_num_dist_list, patch_size_dist_list, win_size, train=1, temp=1, w_de=True):
        for i in range(len(patch_num_dist_list)):
            patch_num_dist = patch_num_dist_list[i]
            patch_size_dist = patch_size_dist_list[i]
            patch_num_dist = repeat(patch_num_dist, 'b n d -> b (n rp) d', rp=win_size // patch_num_dist.shape[1])
            patch_size_dist = repeat(patch_size_dist, 'b p d -> b (rp p) d', rp=win_size // patch_size_dist.shape[1])
            sum_num = torch.sum(patch_num_dist, dim=-1, keepdim=True)
            patch_num_dist = patch_num_dist / sum_num
            sum_size = torch.sum(patch_size_dist, dim=-1, keepdim=True)
            patch_size_dist = patch_size_dist / sum_num
            # 计算双分支损失
            patch_num_loss, patch_size_loss = self.rep_loss(patch_num_dist, patch_size_dist, w_de, train=train, temp=temp)
            # 处理多尺度patch输出
            if i == 0:
                patch_num_loss_all = patch_num_loss
                patch_size_loss_all = patch_size_loss
            else:
                patch_num_loss_all += patch_num_loss
                patch_size_loss_all += patch_size_loss

        return patch_num_loss_all, patch_size_loss_all

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        total_loss=0
        win_size = self.window
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)
            patch_num_loss, patch_size_loss = self.anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                            win_size=win_size, train=1)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)
            lossp = patch_size_loss
            lossq = patch_num_loss
            loss_1.append((lossp).item())
            loss_2.append((lossq).item())
            temp_loss = lossp - lossq
            total_loss += temp_loss

        return total_loss, np.average(loss_1), np.average(loss_2)

    def train_one_epoch(self, data_loader, *args, **kwargs):
        self.model.train()
        total_loss = 0
        for i, (input_data, labels) in enumerate(data_loader):
            self.optimizer.zero_grad()
            input = input_data.float().to(self.device)

            # 多尺度patch列表
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)
            loss = 0.
            cont_loss1, cont_loss2 = self.anomaly_score(patch_num_dist_list, patch_size_mx_list, win_size=self.window, train=1, temp=1)
            cont_loss_1 = cont_loss1 - cont_loss2
            loss -= self.hyper_c * cont_loss_1
            # 计算投影差异损失
            cont_loss12, cont_loss22 = self.anomaly_score(patch_num_mx_list, patch_size_dist_list, win_size=self.window, train=1, temp=1)
            cont_loss_2 = cont_loss12 - cont_loss22
            loss -= self.hyper_c * cont_loss_2
            patch_num_loss, patch_size_loss = self.anomaly_score(patch_num_dist_list, patch_size_dist_list, win_size=self.window, train=1, temp=1)
            # 表征差异损失归一化
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)
            # 关联差异
            loss3 = patch_num_loss - patch_size_loss
            loss -= loss3 * (1 - self.hyper_c)
            # 均方重建损失
            mse = self.loss_func(recx, input)
            loss4 = mse
            loss += loss4 * 10
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        self.model.eval()

        temperature = 1

        '''
        使用训练集及阈值集数据寻找最佳阈值
        attens_energy = []
        cont_beta = self.cont_beta
        mse = nn.MSELoss(reduction='none')
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)
            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list, patch_size_mx_list, window=window, train=0, temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list, win_size=win_size, train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)
            loss3 = patch_size_loss - patch_num_loss
            mse = mse_loss(recx, input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1 - cont_beta)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        attens_energy = []
        print(self.thre_loader.__len__())
        for i, (input_data, labels) in enumerate(self.thre_loader):
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
            loss3 = patch_size_loss - patch_num_loss
            mse_loss_ = mse_loss(recx, input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            # metric1 = -patch_num_loss
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1 - cont_beta)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)
        '''

        project_valid = 0
        test_labels = []
        attens_energy = []
        test_data = []
        for i, (input_data, labels) in enumerate(data_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)
            if project_valid:
                patch_num_loss, patch_size_loss = self.anomaly_score(patch_num_mx_list, patch_size_mx_list, win_size=self.window, train=0, temp=temperature)
            else:
                patch_num_loss, patch_size_loss = self.anomaly_score(patch_num_dist_list, patch_size_dist_list, win_size=self.window, train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)
            loss3 = patch_size_loss - patch_num_loss
            mse_loss_ = self.loss_func(recx, input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * self.hyper_c + metric2 * (1 - self.hyper_c)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            test_data.append(input_data.cpu().numpy().reshape(-1, input_data.shape[-1]))
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        test_data = np.concatenate(test_data, axis=0)
        # 根据异常评分列表，和阈值进行比较，得到相应的异常检测结果
        pred = (test_energy > self.thresh).astype(int)
        gt = test_labels.astype(int)

        # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        # res, thrd, pred2 = self.best_f1(test_energy, gt)
        # 点调整技术
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
        auc_score = roc_auc_score(gt, pred)
        test_loss, vali_loss1, vali_loss2 = self.vali(data_loader)

        return test_loss, zip(metrics,[precision,recall,f_score,auc_score]),pred,gt
