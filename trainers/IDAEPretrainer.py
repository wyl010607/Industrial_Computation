import sys
import warnings

import numpy as np
import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer


class IDAEPretrainer(AbstractTrainer):
    """
    A Trainer subclass for MAE pretraining.
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
        PV_index_list=None,
        OP_index_list=None,
        forward_features=None,
        target_features=None,
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
        PV_index_list : list, optional
            Indices of process variables in the dataset (default is None).
        OP_index_list : list, optional
            Indices of operation variables in the dataset (default is None).
        forward_features : list, optional
            Indices of features to be used as input (default is None).
        target_features : list, optional
            Indices of features to be used as output (default is None).
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
        self.PV_index_list = PV_index_list if PV_index_list is not None else []
        self.OP_index_list = OP_index_list if OP_index_list is not None else []
        self.forward_features = (
            forward_features if forward_features is not None else [0]
        )
        self.target_features = target_features if target_features is not None else [0]

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]

        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def loss_func(self, y_pred, y_true, null_val=np.nan, *args, **kwargs):
        reconstruction_loss = kwargs.get("reconstruction_loss", None)
        # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.
        y_true = torch.where(y_true < 1e-4, torch.zeros_like(y_true), y_true)
        if np.isnan(null_val):
            mask = ~torch.isnan(y_true)
        else:
            mask = y_true != null_val
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(y_pred - y_true)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        loss = torch.mean(loss)
        if reconstruction_loss is not None:
            loss += reconstruction_loss
        return loss

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        tqmd_ = tqdm(data_loader)
        for _, _, pretrain_history_data in tqmd_:
            pretrain_history_data = pretrain_history_data.type(torch.float32).to(
                self.device
            )  # [batch_size, history_len, num_vars, channels]
            batch_size, length, num_nodes, _ = pretrain_history_data.shape
            pretrain_history_data = self.select_input_features(pretrain_history_data)
            # feed forward
            reconstruction, true_value, reconstruction_loss = self.model(history_data=pretrain_history_data)
            loss = self.loss_func(reconstruction, true_value, reconstruction_loss = reconstruction_loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tqmd_.set_description("loss is {:.4f}".format(loss.item()))
            total_loss += loss.item()
        return total_loss / len(data_loader)

    @torch.no_grad()
    def evaluate(self, data_loader, metrics, **kwargs):
        """
        Evaluate the model on the provided dataset.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader containing the evaluation data.
        metrics : list of str
            List of metric names to evaluate the model performance.
        Returns
        -------
        tuple
            train_loss : float
                The average loss on the training data.
            eval_results : list of tuple
                List of tuples of the form (metric_name, metric_value).
            y_pred : numpy.ndarray
                The predicted values.
            y_true : numpy.ndarray
                The true values.
        """
        self.model.eval()

        y_true, y_pred, tol_loss, data_num = [], [], 0, 0
        for _, _, pretrain_history_data in data_loader:
            pretrain_history_data = pretrain_history_data.type(torch.float32).to(
                self.device
            )  # [batch_size, history_len, num_vars, channels]
            # feed forward
            batch_size, length, num_nodes, _ = pretrain_history_data.shape
            pretrain_history_data = self.select_input_features(pretrain_history_data)
            reconstruction_masked_tokens, label_masked_tokens, _ = self.model(
                history_data=pretrain_history_data
            )
            loss = self.loss_func(
                reconstruction_masked_tokens, label_masked_tokens
            ).item()
            tol_loss += loss
            data_num += 1
            y_true.append(label_masked_tokens)
            y_pred.append(reconstruction_masked_tokens)

        # y_pred.shape = [len(data_loader) ,batch_size, time_step, feature_size]
        # to [batch_size * len(data_loader) * time_step, feature_size]
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        eval_results = self.get_eval_result(y_pred, y_true, metrics)
        print("Evaluate result: ", end=" ")
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
        print()
        # reshape y_pred to [batch_size * len(data_loader), time_step, feature_size]
        return (
            tol_loss / data_num,
            zip(metrics, eval_results),
            y_pred,
            y_true,
        )
