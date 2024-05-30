from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer
import torch.nn.functional as F
import time
import numpy as np
from utils.metrics import get_full_err_scores,get_best_performance_data

_device = None

def get_device():
    return _device

device = get_device()


class GDNTrainer(AbstractTrainer):
    """
    A Trainer subclass for sGDN model.
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
        early_stop_patience=5,
        early_stop_min_is_best=True,
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

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = F.mse_loss(y_pred, y_true, reduction='mean')
        return loss

    def train_one_epoch(self, data_loader, *args, **kwargs):
        self.model.train()
        total_loss = 0
        # tqmd_ = tqdm(data_loader)
        train_loss_list = []
        for x, labels, attack_labels, edge_index in data_loader:
            _start = time.time()
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]
            self.optimizer.zero_grad()
            out = self.model(x, edge_index).float().to(device)
            loss = self.loss_func(out, labels)
            loss.backward()
            self.optimizer.step()
            train_loss_list.append(loss.item())
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        self.model.eval()
        y_true, y_pred, tol_loss, data_num = [], [], 0, 0
        test_loss_list = []
        for x, y, labels, edge_index in data_loader:
            x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
            with torch.no_grad():
                predicted = self.model(x, edge_index).float().to(device)
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
            tol_loss += loss.item()
        test_predicted_list = t_test_predicted_list.tolist()
        test_ground_list = t_test_ground_list.tolist()
        test_labels_list = t_test_labels_list.tolist()
        avg_loss = sum(test_loss_list) / len(test_loss_list)
        return avg_loss, test_predicted_list, test_ground_list, test_labels_list

    def get_score(self, test_result, val_result):
        np_test_result = np.array(test_result)
        test_labels = np_test_result[2, :, 0].tolist()
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)
        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        return top1_best_info

    def get_gdn_score(self,test_result, val_result):
        f1, prec, recall, auc = self.get_score(test_result, val_result)
        return f1,prec,recall,auc
