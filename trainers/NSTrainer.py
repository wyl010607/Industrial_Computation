
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

class NSTrainer(AbstractTrainer):


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
        label_len,
        use_amp,
        lradj,
        features,
        patience,
        learning_rate,
        enable_early_stop=False,
        early_stop_patience=5,
        early_stop_min_is_best=True,
        *args,
        **kwargs,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.model_save_path = model_save_path
        self.model_save_dir_path = os.path.dirname(model_save_path)
        self.result_save_dir_path = result_save_dir_path
        self.epoch_now = 0
        self.max_epoch_num = max_epoch_num
        self.enable_early_stop = enable_early_stop
        self.forecast_len=forecast_len
        self.label_len=label_len
        self.use_amp=use_amp
        self.features=features
        self.patience=patience
        self.lradj=lradj
        self.learning_rate=learning_rate
        if self.enable_early_stop:
            self.early_stop = EarlyStop(early_stop_patience, early_stop_min_is_best)
        else:
            self.early_stop = None
        self.device = next(self.model.parameters()).device
        self.min_loss = torch.finfo(torch.float32).max

    def train(
        self,
        train_data_loader,
        eval_data_loader,
        metrics=("mae", "mse", "mape"),
        *args,
        **kwargs,
    ):

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        tmp_state_save_path = os.path.join(self.model_save_dir_path, "temp.pkl")
        epoch_result_list = []
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        for epoch in range(self.epoch_now, self.max_epoch_num):
            print(f"Epoch {epoch} / {self.max_epoch_num}")
            self.save_checkpoint()
            # train
            train_loss = self.train_one_epoch(train_data_loader)  # 训练一个epoch
            self.epoch_now += 1
            print(f"Train loss: {train_loss:.4f}")
            # evaluate
            eval_loss, metrics_evals, _, _ = self.evaluate(eval_data_loader, metrics)
            epoch_result_list.append(
                [epoch, train_loss, eval_loss, list(metrics_evals)]
            )

            early_stopping(eval_loss, self.model, tmp_state_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lradj,self.learning_rate)
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
        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
        epoch_result_json = self._save_epoch_result(epoch_result_list)  # 保存epoch结果
        return epoch_result_json

    def test(self, test_data_loader, metrics=("mae", "mse", "mape"), *args, **kwargs):

        # load best model
        self.model.load_state_dict(torch.load(self.model_save_path))
        # evaluate on test set
        test_loss, metrics_evals, y_pred, y_true = self.evaluate(
            test_data_loader, metrics
        )
        test_result = {"loss": float(test_loss)}
        for metric_name, metric_eval in metrics_evals:
            test_result[metric_name] = metric_eval
        return test_result, y_pred, y_true

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


    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.MSELoss()(y_pred, y_true)
        return loss.mean()


    def train_one_epoch(self, train_loader,*args, **kwargs):

        iter_count = 0
        train_loss = []

        self.model.train()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm.tqdm(train_loader)):
            iter_count += 1
            self.optimizer.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.forecast_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

            # encoder - decoder
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.features == 'MS' else 0
            outputs = outputs[:, -self.forecast_len:, f_dim:]
            batch_y = batch_y[:, -self.forecast_len:, f_dim:].to(self.device)
            loss = self.loss_func(outputs, batch_y)

            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.average(train_loss)

        return train_loss



    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        self.model.eval()
        tol_loss = []
        trues, preds=[], []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.forecast_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.features == 'MS' else 0
                outputs = outputs[:, -self.forecast_len:, f_dim:]
                batch_y = batch_y[:, -self.forecast_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = self.loss_func(pred, true)
                tol_loss.append(loss.item())
                preds.append(pred.numpy())
                trues.append(true.numpy())

        tol_loss = np.average(tol_loss)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        eval_results = self.get_eval_result(preds, trues, metrics)
        print("Evaluate result: ", end=" ")
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
        print()
        self.model.train()
        return (
            tol_loss,
            zip(metrics, eval_results),
            preds,
            trues
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

    def vali(self,  vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.forecast_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.features == 'MS' else 0
                outputs = outputs[:, -self.forecast_len:, f_dim:]
                batch_y = batch_y[:, -self.forecast_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = self.loss_func(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

