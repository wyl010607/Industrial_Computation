import os
import warnings
from shutil import copyfile

import numpy as np
import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer
from torch import optim
from models.SAN.Statistics_prediction import Statistics_prediction
from utils.tools import adjust_learning_rate, EarlyStopping


class CTPGNN_SANTrainer(AbstractTrainer):
    """
    A Trainer subclass for iter multi-step forecasting.
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
            history_len,
            num_route,
            period_len,
            station_type,
            station_lr,
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
        PV_index_list : list, optional
            Indices of process variables in the dataset (default is None).
        OP_index_list : list, optional
            Indices of operation variables in the dataset (default is None).
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
        self.period_len = period_len
        self.station_type = station_type
        self.station_lr = station_lr
        # self._check_model_is_single_step()

        # SAN
        self.statistics_pred = Statistics_prediction(history_len, forecast_len, period_len, num_route, station_type).to(
            self.device)
        self.station_optim = optim.Adam(self.statistics_pred.parameters(), lr=station_lr)

        self.station_pretrain_epoch = 10 if self.station_type == 'adaptive' else 0

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.SmoothL1Loss()(y_pred, y_true)
        return loss.mean()

    def station_loss(self, y, statistics_pred):
        y = y.squeeze(3)
        bs, len, dim = y.shape
        y = y.reshape(bs, -1, self.period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_ture = torch.cat([mean, std], dim=-1)
        loss = torch.nn.MSELoss()(statistics_pred, station_ture) + 0.5/(torch.norm(statistics_pred[:, :, 9:]) + 0.0001)
        return loss

    def train_one_epoch(self, epoch, data_loader):
        self.model.train()
        self.statistics_pred.train()
        total_loss = 0
        tqmd_ = tqdm(data_loader)
        for x, y, stamp in tqmd_:
            x = x.type(torch.float32).to(
                self.device
            )  # [batch_size, history_len, num_vars, channels]
            y = y.type(torch.float32).to(self.device)
            stamp = stamp.type(torch.LongTensor).to(self.device)
            x, statistics_pred = self.statistics_pred.normalize(x.squeeze(3), self.PV_index_list)
            x = x.unsqueeze(3)
            if epoch + 1 <= self.station_pretrain_epoch:
                y = y[:, -self.forecast_len:, 0:, ].to(self.device)
                loss = self.station_loss(y[:, :, self.PV_index_list, :], statistics_pred)

            else:
                sample_pred = torch.zeros_like(y)
                muti_step_pred = torch.zeros_like(y[:, :, self.PV_index_list, :])
                for i in range(self.forecast_len):
                    prediction = self.model(x, stamp, sample_pred)
                   # prediction[torch.isinf(prediction)] = 10
                    #prediction = torch.clamp(prediction, -5, 5)
                    prediction = prediction[:, i: i + 1, :, :]
                    muti_step_pred[:, i: i + 1, :, :] = prediction[
                                                        :, :, self.PV_index_list, :
                                                        ]
                    sample_pred[:, i: i + 1, :, :] = prediction
                    sample_pred[:, i: i + 1, self.OP_index_list, :] = y[
                                                                      :, i: i + 1, self.OP_index_list, :
                                                                      ]
                outputs = self.statistics_pred.de_normalize(muti_step_pred.squeeze(3), statistics_pred)
                loss = self.loss_func(
                    outputs.unsqueeze(3), y[:, :, self.PV_index_list, :]
                )

            self.station_optim.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            if epoch + 1 <= self.station_pretrain_epoch:
                self.station_optim.step()
            else:
                self.optimizer.step()
            tqmd_.set_description("loss is {:.4f}".format(loss.item()))
            total_loss += loss.item()
        return total_loss / len(data_loader)

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
        early_stopping_station_model = EarlyStopping(patience=3, verbose=True)
        # best_model_path = self.model_save_dir_path + '/' + 'checkpoint.pth'
        for epoch in range(self.epoch_now, self.max_epoch_num + self.station_pretrain_epoch):
            print(f"Epoch {epoch} / {self.max_epoch_num + self.station_pretrain_epoch}")
            self.save_checkpoint()

            if epoch == self.station_pretrain_epoch and self.station_type == 'adaptive':
                self.statistics_pred.load_state_dict(torch.load(tmp_state_save_path))
                print('loading pretrained adaptive station model')

            # train
            train_loss = self.train_one_epoch(epoch, train_data_loader)  # 训练一个epoch
            self.epoch_now += 1
            print(f"Train loss: {train_loss:.4f}")
            # evaluate
            eval_loss, metrics_evals, _, _ = self.evaluate(epoch, eval_data_loader, metrics)
            epoch_result_list.append(
                [epoch, train_loss, eval_loss, list(metrics_evals)]
            )

            if epoch + 1 <= self.station_pretrain_epoch:
                early_stopping_station_model(eval_loss, self.statistics_pred, tmp_state_save_path)
                if early_stopping_station_model.early_stop:
                    print("Early stopping")

                adjust_learning_rate(self.station_optim, epoch + 1, 'type1', self.station_lr)

            else:
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
                #adjust_learning_rate(self.optimizer, epoch + 1, 'type1', self.station_lr)
                if self.scheduler is not None:
                    self.scheduler.step()

        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
        epoch_result_json = self._save_epoch_result(epoch_result_list)  # 保存epoch结果
        return epoch_result_json

    @torch.no_grad()
    def evaluate(self, epoch, data_loader, metrics, **kwargs):
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
        self.statistics_pred.eval()
        y_true, y_pred, tol_loss, data_num = [], [], 0, 0
        pred_step = data_loader.dataset.forecast_len
        for x, y, stamp in data_loader:
            x = x.type(torch.float32).to(
                self.device
            )  # [batch_size, history_len, num_vars, channels]
            y = y.type(torch.float32).to(self.device)
            stamp = stamp.type(torch.LongTensor).to(self.device)

            x, statistics_pred = self.statistics_pred.normalize(x.squeeze(3), self.PV_index_list)
            x = x.unsqueeze(3)
            if epoch + 1 <= self.station_pretrain_epoch:
                y = y[:, -self.forecast_len:, 0:, ].to(self.device)
                loss = self.station_loss(y[:, :, self.PV_index_list, :], statistics_pred)

            else:
                sample_pred = torch.zeros_like(y)
                muti_step_pred = torch.zeros_like(y[:, :, self.PV_index_list, :])
                for i in range(self.forecast_len):
                    prediction = self.model(x, stamp, sample_pred)
                    prediction = prediction[:, i: i + 1, :, :]
                    muti_step_pred[:, i: i + 1, :, :] = prediction[
                                                        :, :, self.PV_index_list, :
                                                        ]
                    sample_pred[:, i: i + 1, :, :] = prediction
                    sample_pred[:, i: i + 1, self.OP_index_list, :] = y[
                                                                      :, i: i + 1, self.OP_index_list, :
                                                                      ]
                outputs = self.statistics_pred.de_normalize(muti_step_pred.squeeze(3), statistics_pred)
                loss = self.loss_func(
                    outputs.unsqueeze(3), y[:, :, self.PV_index_list, :]
                )
                y_true.append(y[:, :, self.PV_index_list, :])
                y_pred.append(outputs)
            tol_loss += loss.item()
            data_num += 1

        # y_pred.shape = [len(data_loader) ,batch_size, time_step, feature_size]
        # to [batch_size * len(data_loader) * time_step, feature_size]
        if epoch + 1 > self.station_pretrain_epoch:

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
        else:
            return tol_loss / data_num, {}, 0, 0

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
            self.model.forecast_len = 1

        return self.model

    def test(self, test_data_loader, metrics=("mae", "rmse", "mape"), *args, **kwargs):
        """
        Test the model using the provided test data loader.
        Parameters
        ----------
        test_data_loader : DataLoader
            DataLoader for the test data.
        metrics : tuple of str
            Metrics to evaluate the model performance. Default is ("mae", "rmse", "mape").
        Returns
        -------
        tuple
            test_result : dict
                A dictionary containing the test results.
            y_pred
                Predicted values.
            y_true
                True values.

        """
        # load best model
        self.model.load_state_dict(torch.load(self.model_save_path))
        # evaluate on test set
        test_loss, metrics_evals, y_pred, y_true = self.evaluate(10000,
                                                                 test_data_loader, metrics
                                                                 )
        test_result = {"loss": test_loss}
        for metric_name, metric_eval in metrics_evals:
            test_result[metric_name] = metric_eval
        return test_result, y_pred, y_true
