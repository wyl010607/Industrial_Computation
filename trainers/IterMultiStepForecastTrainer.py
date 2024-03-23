import warnings

import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer


class IterMultiStepForecastTrainer(AbstractTrainer):
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
        self._check_model_is_single_step()

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.SmoothL1Loss()(y_pred, y_true)
        return loss.mean()

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        tqmd_ = tqdm(data_loader)
        for x, y in tqmd_:
            x = x.type(torch.float32).to(
                self.device
            )  # [batch_size, history_len, num_vars, channels]
            y = y.type(torch.float32).to(self.device)
            sample_x = x
            muti_step_pred = torch.zeros_like(y[:, :, self.PV_index_list, :])
            for i in range(y.shape[1]):
                prediction = self.model(sample_x)
                muti_step_pred[:, i : i + 1, :, :] = prediction[
                    :, :, self.PV_index_list, :
                ]
                sample_x = torch.cat((sample_x[:, 1:, :, :], prediction), dim=1)
                sample_x[:, -1:, self.OP_index_list, :] = y[
                    :, i : i + 1, self.OP_index_list, :
                ]
                loss = self.loss_func(
                    muti_step_pred, y[:, :, self.PV_index_list, :]
                )
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
        for x, y in data_loader:
            x = x.type(torch.float32).to(self.device)
            y = y.type(torch.float32).to(self.device)
            muti_step_pred = torch.zeros_like(y[:, :, self.PV_index_list, :])
            sample_x = x
            for i in range(y.shape[1]):
                prediction = self.model(sample_x, iter_step=i).detach()
                muti_step_pred[:, i : i + 1, :, :] = prediction[
                    :, :, self.PV_index_list, :
                ]
                sample_x = torch.cat((sample_x[:, 1:, :, :], prediction), dim=1)
                sample_x[:, -1:, self.OP_index_list] = y[
                    :, i : i + 1, self.OP_index_list, :
                ]
            loss = self.loss_func(
                muti_step_pred, y[:, :, self.PV_index_list, :]
            ).item()
            tol_loss += loss.item()
            data_num += 1
            y_true.append(y[:, :, self.PV_index_list, :])
            y_pred.append(muti_step_pred)

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
