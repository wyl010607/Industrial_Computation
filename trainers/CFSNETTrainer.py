import json
import warnings
import os
from shutil import copyfile
from utils.tools import EarlyStopping, adjust_learning_rate
import numpy as np
import tqdm
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.metrics import metric, cumavg
from einops import rearrange

from trainers.abs import AbstractTrainer

import utils.metrics as metrics_module


class CFSNETTrainer(AbstractTrainer):
    """
    A Trainer subclass for soft sensor data.
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
        forecast_len,
        inverse,
        online_learning,
        n_inner,
        lradj,
        learning_rate,
        enable_early_stop=False,
        early_stop_patience=5,
        early_stop_min_is_best=True,
        PV_index_list=None,
        OP_index_list=None,
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
            *args,
            **kwargs,
        )
        self.PV_index_list = PV_index_list if PV_index_list is not None else []
        self.OP_index_list = OP_index_list if OP_index_list is not None else []
        self.forecast_len = forecast_len
        self.online = online_learning
        self.n_inner = n_inner
        self.early_stop_patience = early_stop_patience
        self.lradj = lradj
        self.learning_rate = learning_rate
        self._check_model_is_single_step()

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.MSELoss()(y_pred, y_true)
        return loss.mean()

    def train_one_epoch(self, data_loader, *args, **kwargs):
        self.model.train()
        total_loss = 0

        for  batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(data_loader):
            self.optimizer.zero_grad()
            batch_x = batch_x.type(torch.float32).to(self.device)
            batch_y = batch_y.type(torch.float32).to(self.device)
            batch_x_mark = batch_x_mark.type(torch.float32).to(self.device)
            batch_y_mark = batch_y_mark.type(torch.float32).to(self.device)
            sample_x = batch_x
            sample_x_mark = batch_x_mark
            muti_step_pred = torch.zeros_like(batch_y[:, :, self.PV_index_list, :])
            for j in range(batch_y.shape[1]):
                pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark[:, j: j + 1, :])
                muti_step_pred[:, j: j + 1, :, :] = pred[
                                                    :, :, self.PV_index_list, :
                                                    ]
                sample_x = torch.cat((sample_x[:, 1:, :, :], pred), dim=1)
                sample_x_mark = torch.cat((sample_x_mark[:, 1:, :], batch_y_mark[:, j: j + 1]), dim=1)
                sample_x[:, -1:, self.OP_index_list, :] = batch_y[
                                                          :, j: j + 1, self.OP_index_list, :
                                                          ]
            loss = self.loss_func(muti_step_pred, batch_y[:, :, self.PV_index_list, :])
            loss.backward()
            self.optimizer.step()
            self.model.store_grad()
            total_loss += loss.item()
        return total_loss / len(data_loader)


    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):

        x = torch.cat([batch_x.float(), batch_x_mark.float().unsqueeze(3)], dim=-2).to(self.device)
        batch_y = batch_y.float()
        outputs = self.model(x)
        return outputs, batch_y


    def train(
        self,
        train_data_loader,
        eval_data_loader,
        metrics=("mae", "rmse", "mape"),
        *args,
        **kwargs,
    ):
        """
        Train the model using the provided training and evaluation data loaders.

        Parameters
        ----------
        train_data_loader : DataLoader
            DataLoader for the training data.
        eval_data_loader : DataLoader
            DataLoader for the evaluation data.
        metrics : tuple of str
            Metrics to evaluate the model performance. Default is ("mae", "rmse", "mape").

        Returns
        -------
        dict
            A dictionary containing training and evaluation results for each epoch.
        """
        tmp_state_save_path = os.path.join(self.model_save_dir_path, "temp.pkl")
        epoch_result_list = []
        early_stopping = EarlyStopping(patience=self.early_stop_patience, verbose=True)
        for epoch in range(self.epoch_now, self.max_epoch_num):
            print(f"Epoch {epoch} / {self.max_epoch_num}")
            self.save_checkpoint()
            # train
            train_loss = self.train_one_epoch(train_data_loader)  # 训练一个epoch
            self.epoch_now += 1
            print(f"Train loss: {train_loss:.4f}")
            # evaluateh
            eval_loss = self.vali(eval_data_loader)
            epoch_result_list.append(
                [epoch, train_loss, eval_loss]
            )
            '''
            # check early stop
            if self.early_stop is not None and self.early_stop.reach_stop_criteria(
                eval_loss
            ):
                self.early_stop.reset()
                break
            
            # save best model
            if eval_loss < self.min_loss:
                self.min_loss = eval_loss
                torch.save(self.model.state_dict(), tmp_state_save_path)
            
            # lr scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            '''
            early_stopping(eval_loss, self.model, tmp_state_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break


            adjust_learning_rate(self.optimizer, epoch + 1, self.lradj, self.learning_rate)

        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
        epoch_result_json = self._save_epoch_result(epoch_result_list)  # 保存epoch结果
        return epoch_result_json


    def vali(self, vali_loader):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.type(torch.float32).to(self.device)
            batch_y = batch_y.type(torch.float32).to(self.device)
            batch_x_mark = batch_x_mark.type(torch.float32).to(self.device)
            batch_y_mark = batch_y_mark.type(torch.float32).to(self.device)
            sample_x = batch_x
            sample_x_mark = batch_x_mark
            muti_step_pred = torch.zeros_like(batch_y[:, :, self.PV_index_list, :])
            for j in range(batch_y.shape[1]):
                pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark[:, j: j + 1, :])
                muti_step_pred[:, j: j + 1, :, :] = pred[
                                                    :, :, self.PV_index_list, :
                                                    ]
                sample_x = torch.cat((sample_x[:, 1:, :, :], pred), dim=1)
                sample_x_mark = torch.cat((sample_x_mark[:, 1:, :], batch_y_mark[:, j: j + 1]), dim=1)
                sample_x[:, -1:, self.OP_index_list, :] = batch_y[
                                                          :, j: j + 1, self.OP_index_list, :
                                                          ]
            loss = self.loss_func(muti_step_pred, batch_y[:, :, self.PV_index_list, :])
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, test_data_loader, metrics=("mae", "rmse", "mape"), *args, **kwargs):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
        if self.online == 'regressor':
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        elif self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False
        y_pred = []
        y_true = []
        mae = []
        mse = []
        mape = []
        pred_step = test_data_loader.dataset.forecast_len
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_data_loader)):
            batch_x = batch_x.type(torch.float32).to(self.device)
            batch_y = batch_y.type(torch.float32).to(self.device)
            batch_x_mark = batch_x_mark.type(torch.float32).to(self.device)
            batch_y_mark = batch_y_mark.type(torch.float32).to(self.device)
            sample_x = batch_x
            sample_x_mark = batch_x_mark
            muti_step_pred = torch.zeros_like(batch_y[:, :, self.PV_index_list, :])
            for j in range(batch_y.shape[1]):
                pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark[:, j: j + 1, :], mode='test')
                muti_step_pred[:, j: j + 1, :, :] = pred[
                                                    :, :, self.PV_index_list, :
                                                    ]
                sample_x = torch.cat((sample_x[:, 1:, :, :], pred), dim=1)
                sample_x_mark = torch.cat((sample_x_mark[:, 1:, :], batch_y_mark[:, j: j + 1]), dim=1)
                sample_x[:, -1:, self.OP_index_list, :] = batch_y[
                                                          :, j: j + 1, self.OP_index_list, :
                                                          ]
            loss = self.loss_func(muti_step_pred, batch_y[:, :, self.PV_index_list, :])
            loss.backward()
            self.optimizer.step()
            self.model.store_grad()
            self.optimizer.zero_grad()

            y_true = self.scaler.inverse_transform(
                batch_y[:, :, self.PV_index_list, :].cpu().detach().numpy().reshape(-1, len(self.PV_index_list)),
                index=self.PV_index_list,
            )
            y_pred = self.scaler.inverse_transform(
                muti_step_pred.cpu().detach().numpy().reshape(-1, len(self.PV_index_list)),
                index=self.PV_index_list,
            )

            mae.append( y_true - y_pred)
            mse.append(((y_true - y_pred) ** 2).mean())

            non_zero_mask = y_true != 0
            y_true_masked = y_true[non_zero_mask]
            y_pred_masked = y_pred[non_zero_mask]
            mape.append(np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100)
            y_true.append(batch_y[:, :, self.PV_index_list, :])
            y_pred.append(muti_step_pred)



        y_true = self.scaler.inverse_transform(
            torch.cat(y_true, dim=0).cpu().detach().numpy().reshape(-1, len(self.PV_index_list)),
            index=self.PV_index_list,
        )
        y_pred = self.scaler.inverse_transform(
            torch.cat(y_pred, dim=0).cpu().detach().numpy().reshape(-1, len(self.PV_index_list)),
            index=self.PV_index_list,
        )
        
        eval_results = self.get_eval_result(y_pred, y_true, metrics)
        print("Evaluate result: ", end=" ")
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
        print()
        # reshape y_pred to [batch_size * len(data_loader), time_step, feature_size]
        test_result = {}
        for metric_name, metric_eval in zip(metrics, eval_results):
            test_result[metric_name] = metric_eval
        #return test_result, y_pred.reshape(-1, pred_step, len(self.PV_index_list), 1), y_true.reshape(-1, pred_step, len(self.PV_index_list), 1),
        return test_result, None, None

    def _save_epoch_result(self, epoch_result_list):
        """
        Save the loss and metrics for each epoch to a json file.
        """
        # save loss and metrics for each epoch to self.result_save_dir/epoch_result.json
        epoch_result = {}
        for epoch, train_loss, eval_loss in epoch_result_list:
            epoch_result[epoch] = {"train_loss": train_loss, "eval_loss": eval_loss}
        with open(
            os.path.join(self.result_save_dir_path, "epoch_result.json"), "w"
        ) as f:
            json.dump(epoch_result, f, indent=4)
        return epoch_result


    def evaluate(self, data_loader, metrics, *args, **kwargs):

        return 0


    def _check_model_is_single_step(self):
        """
        Check if the model is a single step forecasting model. If not, try to set the forecast_len to 1.
        """
        if not hasattr(self.model, "forecast_len"):
            raise AttributeError(
                "The model does not have the attribute 'forecast_len'."
            )
        if self.model.forecast_len != 1:
            warnings.warn(
                "The forecast_len of the model is not 1, it will be set to 1."
            )
            self.model.dim = 37

        return self.model