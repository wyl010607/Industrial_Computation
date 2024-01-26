import warnings
import os
from shutil import copyfile

import numpy as np
import tqdm
from tqdm import tqdm
import torch
import torch.nn.functional as F

from models.TPGNN import predict_stamp
from trainers.abs import AbstractTrainer

import utils.metrics as metrics_module


class TPGNNTrainer(AbstractTrainer):
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
        r,
        max_epoch_num,
        forecast_len,
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
        self.r = r
        self.forecast_len = forecast_len


    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.L1Loss()(y_pred, y_true)
        return loss.mean()

    def train_one_epoch(self, data_loader, epoch, *args, **kwargs):
        self.model.train()
        total_loss = 0
        tqmd_ = tqdm(data_loader)
        for x, stamp, y in tqmd_:
            x, stamp, y = x.cuda(), stamp.cuda(), y.cuda()
            x = x.type(torch.cuda.FloatTensor)
            stamp = stamp.type(torch.cuda.LongTensor)
            y = y.type(torch.cuda.FloatTensor)
            x = x.repeat(2, 1, 1, 1)
            stamp = stamp.repeat(2, 1)
            y = y.repeat(2, 1, 1, 1)

            y_pred, loss = self.model(x, stamp, y, epoch)
            bs = y.shape[0]
            y_pred1 = y_pred[:bs // 2, :, :, :]
            y_pred2 = y_pred[bs // 2:, :, :, :]
            r_loss = F.l1_loss(y_pred1, y_pred2)
            r_loss = r_loss * self.r
            loss = loss + r_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tqmd_.set_description("loss is {:.4f}".format(loss.item()))
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        self.model.eval()
        y_true_ls, y_pred_ls, tol_loss, data_num = [], [], 0, 0

        length = self.forecast_len // 3

        with torch.no_grad():
            mae = [[] for _ in range(length)]
            mape = [[] for _ in range(length)]
            mse = [[] for _ in range(length)]
            MAE, MAPE, RMSE = [0.0] * length, [0.0] * length, [0.0] * length

            for x, stamp, y in data_loader:
                x, stamp, y = x.cuda(), stamp.cuda(), y.cuda()
                x = x.type(torch.cuda.FloatTensor)
                stamp = stamp.type(torch.cuda.LongTensor)
                y = y.type(torch.cuda.FloatTensor)
                num_route = y.shape[2]
                y_pred = predict_stamp(self.model, x, stamp, y)
                loss = self.loss_func(y_pred, y)
                tol_loss += loss.item()
                data_num += 1

                y_pred = y_pred.permute(0, 3, 1, 2).reshape(-1, num_route)
                y_true_ls.append(y)
                y_pred_ls.append(y_pred)

                y_pred = self.scaler.inverse_transform(
                    y_pred.cpu().numpy()).reshape(-1, 1, self.forecast_len, num_route)
                y = self.scaler.inverse_transform(
                    y.permute(0, 3, 1, 2).reshape(-1, num_route).cpu().numpy()).reshape(-1, 1, self.forecast_len, num_route)

                for i in range(length):
                    y_pred_select = y_pred[:, :, 3 * i + 2, :].reshape(-1)
                    y_select = y[:, :, 3 * i + 2, :].reshape(-1)

                    y_pred_select = torch.from_numpy(y_pred_select)
                    y_select = torch.from_numpy(y_select)
                    mae[i] += metrics_module.masked_mae(y_pred_select,
                                                        y_select, 0.0).numpy().tolist()
                    mape[i] += metrics_module.masked_mape(y_pred_select,
                                                          y_select, 0.0).numpy().tolist()
                    mse[i] += metrics_module.masked_mse(y_pred_select,
                                                        y_select, 0.0).numpy().tolist()


            for j in range(length):
                MAE[j] = round(np.array(mae[j]).mean(), 4)
                MAPE[j] = round(np.array(mape[j]).mean(), 4)
                RMSE[j] = round(np.sqrt(np.array(mse[j]).mean()), 4)

            y_true_ls = self.scaler.inverse_transform(
                torch.cat(y_true_ls, dim=0).cpu().numpy().reshape(-1, num_route),
            )
            y_pred_ls = self.scaler.inverse_transform(
                torch.cat(y_pred_ls, dim=0).cpu().numpy().reshape(-1, num_route),
            )


        eval_results = []
        eval_results.append(MAE)
        eval_results.append(RMSE)
        eval_results.append(MAPE)
        # reshape y_pred to [batch_size * len(data_loader), feature_size]
        print("Evaluate result: ", end=" ")
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {}".format(metric_name.upper(), eval_ret), end="  ")
        print()
        return (
            tol_loss / data_num,
            zip(metrics, eval_results),
            y_pred_ls.reshape(-1, self.forecast_len, num_route, 1),
            y_true_ls.reshape(-1, self.forecast_len, num_route, 1),
        )

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

        for epoch in range(self.epoch_now+1, self.max_epoch_num+1):
            print(f"Epoch {epoch} / {self.max_epoch_num}")
            self.save_checkpoint()
            # train
            train_loss = self.train_one_epoch(train_data_loader, epoch)  # 训练一个epoch
            self.epoch_now += 1
            print(f"Train loss: {train_loss:.4f}")
            # evaluate
            eval_loss, metrics_evals, _, _ = self.evaluate(eval_data_loader, metrics)
            epoch_result_list.append(
                [epoch, train_loss, eval_loss, list(metrics_evals)]
            )

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

        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
        epoch_result_json = self._save_epoch_result(epoch_result_list)  # 保存epoch结果
        return epoch_result_json

