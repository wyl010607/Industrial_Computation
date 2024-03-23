import warnings

import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer


class SoftSensorTrainer(AbstractTrainer):
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
        self.input_index_list = input_index_list
        self.output_index_list = output_index_list

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = torch.nn.SmoothL1Loss()(y_pred, y_true)
        return loss.mean()

    def train_one_epoch(self, data_loader, *args, **kwargs):
        self.model.train()
        total_loss = 0
        tqmd_ = tqdm(data_loader)
        for x, y in tqmd_:
            x = x.type(torch.float32).to(
                self.device
            )  # [batch_size, history_len, num_vars, channels]
            pred = self.model(x)
            y = y.type(torch.float32).to(self.device)
            loss = self.model_loss_func(pred, y)
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
            x = x.type(torch.float32).to(
                self.device
            )
            pred = self.model(x)
            y = y.type(torch.float32).to(self.device)
            loss = self.model_loss_func(pred, y).item()
            tol_loss += loss
            data_num += 1
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
        y_true = self.scaler.inverse_transform(
            torch.cat(y_true, dim=0).cpu().numpy().reshape(-1, len(self.output_index_list)),
            index=self.output_index_list,
        )
        y_pred = self.scaler.inverse_transform(
            torch.cat(y_pred, dim=0).cpu().numpy().reshape(-1, len(self.output_index_list)),
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
