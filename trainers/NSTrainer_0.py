
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from trainers.abs import AbstractTrainer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from models.TPGNN import predict_stamp
import tqdm
warnings.filterwarnings('ignore')
import json
import os
from shutil import copyfile

import torch
from abc import ABC, abstractmethod

from utils.early_stop import EarlyStop
import utils.metrics as metrics_module

class NSTrainer_0(AbstractTrainer):

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
        self.label_len=kwargs.get("label_len")
        self.learning_rate = learning_rate
        self._check_model_is_single_step()

    def train(
            self,
            train_data_loader,
            eval_data_loader,
            metrics=("mae", "mse", "mape"),
            *args,
            **kwargs,
    ):
        print("self.label_len=",self.label_len)
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
            eval_loss, metrics_evals, _, _ = self.evaluate(eval_data_loader, metrics)
            epoch_result_list.append(
                [epoch, train_loss, eval_loss, list(metrics_evals)]
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
    '''
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
                pred, true = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark[:, j: j + 1, :],
                                                     mode='test')
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

            mae.append(y_true - y_pred)
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

        return test_result, y_pred.reshape(-1, pred_step, len(self.PV_index_list), 1), y_true.reshape(-1, pred_step, len(self.PV_index_list), 1),
        #return test_result, None, None
    '''

    def save_checkpoint(self, filename="checkpoint.pth"):
        """
        Save the current training state as a checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "epoch": self.epoch_now,
            "min_loss": self.min_loss,
        }
        torch.save(checkpoint, os.path.join(self.model_save_dir_path, filename))

    def load_checkpoint(self, filename="checkpoint.pth"):
        """
        Load training state from a checkpoint.
        """

        filepath = os.path.join(self.model_save_dir_path, filename)
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.epoch_now = checkpoint.get("epoch", 1)
            self.min_loss = checkpoint.get("min_loss", torch.finfo(torch.float32).max)
        else:
            raise FileNotFoundError(f"No checkpoint found at '{filepath}'")
    '''
    def _save_epoch_result(self, epoch_result_list):
        """
        Save the loss and metrics for each epoch to a json file.
        """
        # save loss and metrics for each epoch to self.result_save_dir/epoch_result.json
        epoch_result = {}
        for epoch, train_loss, eval_loss, metrics_evals in epoch_result_list:
            epoch_result[epoch] = {"train_loss": float(train_loss), "eval_loss": float(eval_loss)}
            for metric_name, metric_eval in metrics_evals:
                epoch_result[epoch][metric_name] = metric_eval
        with open(
            os.path.join(self.result_save_dir_path, "epoch_result.json"), "w"
        ) as f:
            json.dump(epoch_result, f, indent=4)
        return epoch_result
    '''

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.MSELoss()(y_pred, y_true)
        return loss.mean()


    def train_one_epoch(self, data_loader, *args, **kwargs):
        self.model.train()
        total_loss = 0

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm.tqdm(data_loader)):
            #print("sssssssssssssssssssssssssssssssssssssssssssssssss")
            self.optimizer.zero_grad()
            #print("batch_x.shape",batch_x.shape)
            #print("batch_y.shape",batch_y.shape)
            batch_x = batch_x.type(torch.float32).to(self.device)
            batch_y = batch_y.type(torch.float32).to(self.device)
            batch_x_mark = batch_x_mark.type(torch.float32).to(self.device)
            batch_y_mark = batch_y_mark.type(torch.float32).to(self.device)
            sample_x = batch_x
            sample_x_mark = batch_x_mark
            muti_step_pred = torch.zeros_like(batch_y[:, :, self.PV_index_list, :])
            #print("batch_x.shape",batch_x.shape)
            #print("batch_y.shape",batch_y.shape)
            dec_inp = torch.zeros_like(batch_y[:, -self.forecast_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
            #print("dec_inp.shape",dec_inp.shape)

            for j in range(batch_y.shape[1]):
                pred = self.model(batch_x, batch_x_mark, dec_inp,batch_y_mark[:, j: j + 1, :])
                #print(pred.shape)
                #print("qqqqqqqqqqqqqqqqqqqqqqqqq")
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
            total_loss += loss.item()
            #print("total_loss:",total_loss)
        return total_loss / len(data_loader)

    @torch.no_grad()
    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        self.model.eval()
        tol_loss = 0.0
        trues, preds=[], []
        data_num = 0
        pred_step = data_loader.dataset.forecast_len
        #print("fffffffffffffffffffffffffffffffffffffff")
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            #print("llllllllllllllllllllllllllllllllllllllllll")
            batch_x = batch_x.type(torch.float32).to(self.device)
            batch_y = batch_y.type(torch.float32).to(self.device)
            batch_x_mark = batch_x_mark.type(torch.float32).to(self.device)
            batch_y_mark = batch_y_mark.type(torch.float32).to(self.device)
            sample_x = batch_x
            sample_x_mark = batch_x_mark
            muti_step_pred = torch.zeros_like(batch_y[:, :, self.PV_index_list, :])
            dec_inp = torch.zeros_like(batch_y[:, -self.forecast_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
            #print(batch_y.shape)
            for j in range(batch_y.shape[1]):
                pred = self.model(batch_x, batch_x_mark,dec_inp, batch_y_mark[:, j: j + 1, :])
                muti_step_pred[:, j: j + 1, :, :] = pred[
                                                        :, :, self.PV_index_list, :
                                                        ]
                sample_x = torch.cat((sample_x[:, 1:, :, :], pred), dim=1)
                sample_x_mark = torch.cat((sample_x_mark[:, 1:, :], batch_y_mark[:, j: j + 1]), dim=1)
                sample_x[:, -1:, self.OP_index_list, :] = batch_y[
                                                              :, j: j + 1, self.OP_index_list, :
                                                              ]
            #pred = muti_step_pred.detach().cpu()
            #true = batch_y[:, :, self.PV_index_list, :].detach().cpu()
            loss = self.loss_func(muti_step_pred, batch_y[:, :, self.PV_index_list, :])
            #loss = self.loss_func(pred, true)
            data_num += 1
            tol_loss += loss.item()
            #print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
            preds.append(muti_step_pred)
            trues.append(batch_y[:, :, self.PV_index_list, :])
        #print(preds)
        y_true = self.scaler.inverse_transform(
            torch.cat(preds, dim=0).cpu().numpy().reshape(-1, len(self.PV_index_list)),
            index=self.PV_index_list,
        )
        y_pred = self.scaler.inverse_transform(
            torch.cat(trues, dim=0).cpu().numpy().reshape(-1, len(self.PV_index_list)),
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


    def get_eval_result(self,y_pred, y_true, metrics=("mae", "mse", "mape")):
        """
        Compute evaluation metrics for the given predictions and true values.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_true
            True values.
        metrics : tuple of str
            Metrics to be computed.

        Returns
        -------
        list
            List of computed metric values.
        """
        eval_results = []
        for metric_name in metrics:
            eval_func_name = "get_{}".format(metric_name)
            eval_func = getattr(metrics_module, eval_func_name, None)

            if eval_func is None:
                raise AttributeError(
                    f"Function '{eval_func_name}' not found in 'utils.metrics'."
                )

            result = float(eval_func(y_pred, y_true))
            eval_results.append(result)

        return eval_results
    '''
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
                pred, true = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark[:, j: j + 1, :])
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
    '''

    def _check_model_is_single_step(self):
        """
        Check if the model is a single step forecasting model. If not, try to set the forecast_len to 1.
        """
        #print("self.model.forecast_len=",self.model.forecast_len)
        if not hasattr(self.model, "forecast_len"):
            raise AttributeError(
                "The model does not have the attribute 'forecast_len'."
            )
        if self.model.forecast_len != 1:
            warnings.warn(
                "The forecast_len of the model is not 1, it will be set to 1."
            )
            self.model.forecast_len = 1

        return self.model

