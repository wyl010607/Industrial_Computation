import json
import os
import sys
import warnings

import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer


class STDMAETrainer(AbstractTrainer):
    """
    A Trainer subclass for STDMAE training.
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
        pretrain_model_name=None,
        pretrain_model_params=None,
        pretrainer_params=None,
        skip_pretrain=False,
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
        pretrain_model_name : str, optional
            Name of the pretrain model (default is None).
        pretrain_model_params : dict, optional
            Parameters for pretrain model (default is None).
        pretrainer_params : dict, optional
            Parameters for pretrainer (default is None).
        skip_pretrain : bool, optional
            Whether to skip pretraining (default is False).
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
        self.pretrain_s_model_save_path = self.model_save_path + "_pretrain_s_model"
        self.pretrain_s_model_result_save_dir_path = (
            self.result_save_dir_path + "/pretrain_s_model"
        )
        self.pretrain_t_model_save_path = self.model_save_path + "_pretrain_t_model"
        self.pretrain_t_model_result_save_dir_path = (
            self.result_save_dir_path + "/pretrain_t_model"
        )
        self.pretrain_model_name = pretrain_model_name
        self.pretrain_model_params = (
            pretrain_model_params if pretrain_model_params is not None else {}
        )
        self.pretrainer_params = (
            pretrainer_params if pretrainer_params is not None else {}
        )
        self.skip_pretrain = skip_pretrain
        if self.skip_pretrain:
            warnings.warn("Skip pretraining.")
        else:
            self.pretrain_s_model, self.pretrain_t_model = self._init_pretrain_model(
                self.pretrain_model_name, self.pretrain_model_params
            )
            (
                self.pretrain_s_model_trainer,
                self.pretrain_t_model_trainer,
            ) = self._init_pretrainer(self.pretrainer_params)

    def _init_pretrain_model(self, pretrain_model_name, pretrain_model_params):
        # model
        model_class = getattr(sys.modules["models"], pretrain_model_name)
        s_model = model_class(**pretrain_model_params["s_model_params"])
        t_model = model_class(**pretrain_model_params["t_model_params"])
        s_model.to(self.device)
        t_model.to(self.device)
        return s_model, t_model

    def _init_pretrainer(self, pretrainer_params):
        # Optimizer
        optimizer_class = getattr(
            sys.modules["torch.optim"], pretrainer_params["optimizer_name"]
        )
        self.pretrain_s_model_optimizer = optimizer_class(
            self.pretrain_s_model.parameters(),
            **pretrainer_params["optimizer_params"]["s_model_params"],
        )
        self.pretrain_t_model_optimizer = optimizer_class(
            self.pretrain_t_model.parameters(),
            **pretrainer_params["optimizer_params"]["t_model_params"],
        )
        # scheduler
        scheduler_class = getattr(
            sys.modules["torch.optim.lr_scheduler"], pretrainer_params["scheduler_name"]
        )
        self.pretrain_s_model_scheduler = scheduler_class(
            self.pretrain_s_model_optimizer,
            **pretrainer_params["scheduler_params"]["s_model_params"],
        )
        self.pretrain_t_model_scheduler = scheduler_class(
            self.pretrain_t_model_optimizer,
            **pretrainer_params["scheduler_params"]["t_model_params"],
        )

        # trainer
        trainer_class = getattr(
            sys.modules["trainers"], pretrainer_params["trainer_name"]
        )
        self.pretrain_s_model_trainer = trainer_class(
            self.pretrain_s_model,
            self.pretrain_s_model_optimizer,
            self.pretrain_s_model_scheduler,
            self.scaler,
            self.pretrain_s_model_save_path,
            self.pretrain_s_model_result_save_dir_path,
            **pretrainer_params["trainer_params"],
        )
        self.pretrain_t_model_trainer = trainer_class(
            self.pretrain_t_model,
            self.pretrain_t_model_optimizer,
            self.pretrain_t_model_scheduler,
            self.scaler,
            self.pretrain_t_model_save_path,
            self.pretrain_t_model_result_save_dir_path,
            **pretrainer_params["trainer_params"],
        )
        return self.pretrain_s_model_trainer, self.pretrain_t_model_trainer

    def _pretrain(self, train_data_loader, eval_data_loader):
        print("Start pretraining.")
        self.pretrain_s_model_trainer.train(train_data_loader,eval_data_loader)
        self.pretrain_t_model_trainer.train(train_data_loader,eval_data_loader)
        print("Pretraining finished.")

    def train(
        self,
        train_data_loader,
        eval_data_loader,
        metrics=("mae", "rmse", "mape"),
        *args,
        **kwargs,
    ):
        if not self.skip_pretrain:
            self._pretrain(train_data_loader, eval_data_loader)
        super().train(train_data_loader, eval_data_loader, metrics, *args, **kwargs)

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.SmoothL1Loss()(y_pred, y_true)
        return loss.mean()

    def train_one_epoch(self, data_loader):
        if self.epoch_now == 0:
            self.model.load_pretrained_model(
                self.pretrain_s_model_save_path, self.pretrain_t_model_save_path
            )
        self.model.train()
        total_loss = 0
        tqmd_ = tqdm(data_loader)
        for history_data, future_data, long_history_data in tqmd_:
            # 直接多步预测
            history_data = history_data.type(torch.float32).to(
                self.device
            )  # [batch_size, history_len, num_vars, channels]
            future_data = future_data.type(torch.float32).to(self.device)
            long_history_data = long_history_data.type(torch.float32).to(self.device)
            prediction = self.model(history_data, long_history_data)[:, :, self.PV_index_list, :]
            loss = self.loss_func(prediction, future_data[:, :, self.PV_index_list, :])
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
        pred_step = data_loader.dataset.forecast_len
        for history_data, future_data, long_history_data in data_loader:
            # 直接多步预测
            history_data = history_data.type(torch.float32).to(
                self.device
            )  # [batch_size, history_len, num_vars, channels]
            future_data = future_data.type(torch.float32).to(self.device)
            long_history_data = long_history_data.type(torch.float32).to(self.device)
            prediction = self.model(history_data, long_history_data)[:, :, self.PV_index_list, :]
            loss = self.loss_func(prediction, future_data[:, :, self.PV_index_list, :])
            tol_loss += loss.item()
            data_num += 1
            y_true.append(future_data[:, :, self.PV_index_list, :])
            y_pred.append(prediction)

        # y_pred.shape = [len(data_loader) ,batch_size, time_step, feature_size]
        # to [batch_size * len(data_loader) * time_step, feature_size]
        y_true = self.scaler.inverse_transform(
            torch.cat(y_true, dim=0).cpu().numpy().reshape(-1, len(self.PV_index_list)),
            index=self.PV_index_list,
        )
        y_pred = self.scaler.inverse_transform(
            torch.cat(y_pred, dim=0).cpu().numpy().reshape(-1, len(self.PV_index_list)),
            index=self.PV_index_list,
        )
        eval_results = self.get_eval_result(y_pred, y_true, metrics)
        print("Evaluate result: ", end=" ")
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
        print()
        # reshape y_pred to [batch_size * len(data_loader), time_step, feature_size]
        return (
            tol_loss / data_num,
            zip(metrics, eval_results),
            y_pred.reshape(-1, pred_step, len(self.PV_index_list), 1),
            y_true.reshape(-1, pred_step, len(self.PV_index_list), 1),
        )

