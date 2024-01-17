import json
import os
import sys
import warnings

import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer


class SAETrainer(AbstractTrainer):
    """
    A Trainer subclass for stacked autoencoder.
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
        train_AE_epochs,
        train_AE_optimizer_name,
        train_AE_optimizer_params,
        enable_early_stop=False,
        early_stop_patience=5,
        early_stop_min_is_best=True,
        input_index_list=None,
        output_index_list=None,
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
        train_AE_epochs : int
            The number of epochs for training each AE.
        train_AE_batch_size : int
            The batch size for training each AE.
        train_AE_optimizer_name : str
            The name of the optimizer for training each AE.
        train_AE_optimizer_params : dict
            The parameters of the optimizer for training each AE.
        enable_early_stop : bool, optional
            Whether to enable early stopping (default is False).
        early_stop_patience : int, optional
            Number of epochs with no improvement after which training will be stopped (default is 5).
        early_stop_min_is_best : bool, optional
            Flag to indicate if lower values of loss indicate better performance (default is True).
        input_index_list : list, optional
            Indices of input variables in the dataset (default is None).
        output_index_list : list, optional
            Indices of output variables in the dataset (default is None).
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
            input_index_list,
            output_index_list,
            *args,
            **kwargs,
        )
        self.input_index_list = input_index_list
        self.output_index_list = output_index_list
        self.train_AE_epochs = train_AE_epochs
        self.train_AE_optimizer_name = train_AE_optimizer_name
        self.train_AE_optimizer_params = train_AE_optimizer_params

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.SmoothL1Loss()(y_pred, y_true)
        return loss.mean()

    def _train_AE(
        self,
        model,
        pretrain_data_loader,
        pretrain_layer_numero,
        pretrain_epoch,
        pretrain_optimizer,
        pretrain_optimizer_params,
    ):
        # freeze the parameters of the previous layers
        for layer in range(pretrain_layer_numero):
            for param in model.SAE[layer].parameters():
                param.requires_grad = False

        # reset data loader batch size
        pretrain_optimizer_class = getattr(
            sys.modules["torch.optim"], pretrain_optimizer
        )
        pretrain_optimizer = pretrain_optimizer_class(
            model.SAE[pretrain_layer_numero].parameters(), **pretrain_optimizer_params
        )
        pretrain_loss_func = torch.nn.MSELoss()
        model.train()
        # pretrain result list
        pretrain_epoch_result_list = []
        # use tqdm to show training progress
        for epoch in tqdm(range(pretrain_epoch)):
            epoch_loss = 0
            for x, y in pretrain_data_loader:
                x = x.to(self.device)
                hidden, hidden_reconst = model(
                    x, is_pretrain=True, pretrain_layer_numero=pretrain_layer_numero
                )
                loss = pretrain_loss_func(hidden_reconst, hidden)
                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()
                epoch_loss += loss.item()
            pretrain_epoch_result_list.append(
                {"epoch": epoch, "loss": epoch_loss / len(pretrain_data_loader)}
            )
        # unfreeze the parameters of the previous layers
        for layer in model.SAE:
            for param in layer.parameters():
                param.requires_grad = True
        return model, pretrain_epoch_result_list

    def _save_pretrain_result(self, pretrain_result_list):
        """
        Save the pretrain result to json file.
        """
        pretrain_result_json = dict()
        for i in range(len(pretrain_result_list)):
            pretrain_result_json["AE layer {}".format(i + 1)] = pretrain_result_list[i]
        save_path = os.path.join(self.result_save_dir_path, "pretrain_result.json")
        with open(save_path, "w") as f:
            json.dump(pretrain_result_json, f, indent=4)

    def train_one_epoch(self, data_loader, *args, **kwargs):
        # pretrain

        if self.epoch_now == 0:
            pretrain_result_list = []
            for i in range(self.model.AElength):
                print("Pretrain AE layer {}".format(i + 1))
                self.model, pretrain_epoch_result_list = self._train_AE(
                    self.model,
                    data_loader,
                    i,
                    self.train_AE_epochs,
                    self.train_AE_optimizer_name,
                    self.train_AE_optimizer_params,
                )
                pretrain_result_list.append(pretrain_epoch_result_list)
                print("Pretrain AE layer {} finished!".format(i + 1))
            self._save_pretrain_result(pretrain_result_list)
        # fine-tuning
        self.model.train()
        total_loss = 0
        tqmd_ = tqdm(data_loader)
        for x, y in tqmd_:
            x = x.type(torch.float32).to(
                self.device
            )  # [batch_size, history_len, num_vars, channels]
            pred = self.model(x)
            y = y.type(torch.float32).to(self.device)
            loss = self.loss_func(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tqmd_.set_description("loss is {:.4f}".format(loss.item()))
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        self.model.eval()
        y_true, y_pred, tol_loss, data_num = [], [], 0, 0
        for x, y in data_loader:
            x = x.type(torch.float32).to(self.device)
            pred = self.model(x)
            y = y.type(torch.float32).to(self.device)
            loss = self.loss_func(pred, y).item()
            tol_loss += loss
            data_num += 1
            y_true.append(y.detach())
            y_pred.append(pred.detach())
        y_true = self.scaler.inverse_transform(
            torch.cat(y_true, dim=0)
            .cpu()
            .numpy()
            .reshape(-1, len(self.output_index_list)),
            index=self.output_index_list,
        )
        y_pred = self.scaler.inverse_transform(
            torch.cat(y_pred, dim=0)
            .cpu()
            .numpy()
            .reshape(-1, len(self.output_index_list)),
            index=self.output_index_list,
        )
        eval_results = self.get_eval_result(y_pred, y_true, metrics)
        print("Evaluate result: ", end=" ")
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
        print()
        # reshape y_pred to [batch_size * len(data_loader), feature_size]
        return (
            tol_loss / data_num,
            zip(metrics, eval_results),
            y_pred.reshape(-1, len(self.output_index_list)),
            y_true.reshape(-1, len(self.output_index_list)),
        )
