from tqdm import tqdm
import torch
import time
import numpy as np
from trainers.abs import AbstractTrainer
from utils.metrics import get_performance


class DCTrainer(AbstractTrainer):
    """
    A Trainer subclass for DCdetector Model.
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
        index,
        enable_early_stop=False,
        early_stop_patience=5,
        early_stop_min_is_best=True,
        window=100,
        thresh=0.7,
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
        self.index = index

    def kl_loss(self, p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    def adjust_learning_rate(self, epoch, lr_):
        lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            self.update_trainer_params["lr"] = lr

    '''
    def combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores):
        events_pred = convert_vector_to_events(y_test)
        events_gt = convert_vector_to_events(pred_labels)
        Trange = (0, len(y_test))
        affiliation = pr_from_events(events_pred, events_gt, Trange)
        true_events = get_events(y_test)
        pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(y_test, pred_labels)
        MCC_score = MCC(y_test, pred_labels)
        vus_results = get_range_vus_roc(y_test, pred_labels, 100)  # default slidingWindow = 100

        score_list_simple = {
            "pa_accuracy": pa_accuracy,
            "pa_precision": pa_precision,
            "pa_recall": pa_recall,
            "pa_f_score": pa_f_score,
            "MCC_score": MCC_score,
            "Affiliation precision": affiliation['precision'],
            "Affiliation recall": affiliation['recall'],
            "R_AUC_ROC": vus_results["R_AUC_ROC"],
            "R_AUC_PR": vus_results["R_AUC_PR"],
            "VUS_ROC": vus_results["VUS_ROC"],
            "VUS_PR": vus_results["VUS_PR"]
        }
        # return score_list, score_list_simple
        return score_list_simple
    '''

    def vali(self, vali_loader):
        self.model.eval()
        loss = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(self.kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach())) + torch.mean(
                    self.kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    self.kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    self.kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            loss.append((prior_loss - series_loss).item())

        return np.average(loss)

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.SmoothL1Loss()(y_pred, y_true)
        return loss.mean()

    def train_one_epoch(self, data_loader, *args, **kwargs):
        self.model.train()
        total_loss = 0
        loss_list =[]
        tqmd_ = tqdm(data_loader)

        for input_data, labels in enumerate(data_loader):
            self.optimizer.zero_grad()
            input = input_data.float().to(self.device)
            series, prior = self.model(input)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(self.kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.window)).detach())) + torch.mean(
                    self.kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.window)).detach(),
                               series[u])))
                prior_loss += (torch.mean(self.kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.window)),
                    series[u].detach())) + torch.mean(
                    self.kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.window)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            # 构造表征差异
            loss = prior_loss - series_loss
            loss.backward()
            self.optimizer.step()
            loss_list.append((prior_loss - series_loss).item())
        total_loss = np.average(loss_list)
        return total_loss

    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        self.model.eval()

        '''
        阈值微调，涉及到thre_data,无法很好地迁移到trainer中
        attens_energy = []
        for input_data in data_loader:  # threhold
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.seq_len)).detach())
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.seq_len)),
                        series[u].detach())
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.seq_len)).detach())
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.seq_len)),
                        series[u].detach())

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        thresh = np.percentile(test_energy, 100 - self.threshold)
        '''

        temperature = 10
        # evaluation on the test set
        test_labels = []
        attens_energy = []
        for input_data, labels in enumerate(data_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = self.kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.window)).detach()) * temperature
                    prior_loss = self.kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.window)),
                        series[u].detach()) * temperature
                else:
                    series_loss += self.kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.window)).detach()) * temperature
                    prior_loss += self.kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.window)),series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
        attens_energy = np.concatenate(attens_energy, axis=0) # 异常评分计算
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > self.thresh).astype(int)
        gt = test_labels.astype(int)

        # scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)

        # 点调整技术的具体实现
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

        test_r = get_performance(pred, gt)
        test_loss = self.vali(data_loader)

        return test_loss, zip(metrics,test_r), pred, gt
