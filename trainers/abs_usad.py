import json
import os
import time
import numpy as np
from shutil import copyfile
import torch
from abc import ABC, abstractmethod
from utils.early_stop import EarlyStop, EarlyStopping
import utils.metrics as metrics_module
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class AbstractTrainer(ABC):
    """
    Abstract base class for training machine learning models.

    Methods
    -------
    train(train_data_loader, eval_data_loader, metrics=("mae", "rmse", "mape"), *args, **kwargs)
        Train the model with given data loaders and metrics.
    test(test_data_loader, metrics=("mae", "rmse", "mape"), *args, **kwargs)
        Test the model using the test data loader and specified metrics.
    save_checkpoint(filename="checkpoint.pth")
        Save the current training state as a checkpoint.
    load_checkpoint(filename="checkpoint.pth")
        Load training state from a checkpoint.
    """

    def __init__(
        self,
        model,
        scheduler,
        scaler,
        model_save_path,
        result_save_dir_path,
        max_epoch_num,
        # enable_early_stop=False,
        early_stop_patience=15,
        # early_stop_min_is_best=True,
        *args,
        **kwargs,
    ):
        self.model = model
        self.scaler = scaler
        self.scheduler = scheduler
        self.model_save_path = model_save_path
        self.model_save_dir_path = os.path.dirname(model_save_path)
        self.result_save_dir_path = result_save_dir_path
        self.epoch_now = 0
        self.max_epoch_num = max_epoch_num
        self.early_stop_patience = early_stop_patience
        self.device = next(self.model.parameters()).device

    def train(
        self,
        train_loader,
        val_loader,
        metrics=("prec", "rec", "f1", "auc"),
        *args,
        **kwargs,
    ):
        tmp_state_save_path = os.path.join(self.model_save_dir_path, "temp.pkl")
        epoch_result_list = []

        for epoch in range(self.epoch_now, self.max_epoch_num):
            print(f"Epoch {epoch} / {self.max_epoch_num}")
            self.save_checkpoint()
            # train
            time_now = time.time()
            train_loss1, train_loss2 = self.train_one_epoch(train_loader, self.epoch_now, time_now)  # 训练一个epoch
            self.epoch_now += 1
            # evaluate
            print(f"验证阶段:Epoch {epoch}")
            result = self.vali(val_loader, epoch + 1)
            self.model.epoch_end(epoch, result)

            torch.save(self.model.state_dict(), tmp_state_save_path)

            # lr scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
        epoch_result_json = self._save_epoch_result(epoch_result_list)  # 保存epoch结果
        return epoch_result_json

    def vali(self, val_loader, n):
        outputs = [self.model.validation_step(to_device(batch, self.device), n) for [batch] in val_loader]
        return self.model.validation_epoch_end(outputs)

    def test(self, test_loader, labels, metrics=("prec", "rec", "f1", "auc"), *args, **kwargs):
        # load best model
        self.model.load_state_dict(torch.load(self.model_save_path))
        # evaluate on test set
        print("模型测试：")
        self.epoch_now = self.epoch_now+1
        result = self.evaluate(test_loader, metrics)
        y_pred = np.concatenate([torch.stack(result[:-1]).flatten().detach().cpu().numpy(),
                        result[-1].flatten().detach().cpu().numpy()])
        y_test = [1.0 if (np.sum(window) > 0) else 0 for window in labels]
        precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
        auc_score = roc_auc_score(y_test, y_pred)
        test_result = {"precision": precision, "recall":recall, "f1_score":f_score, "auc_score":auc_score}
        return y_test, y_pred, test_result

    def save_checkpoint(self, filename="checkpoint.pth"):
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
        epoch_result = {}
        for epoch, train_loss, val_loss, val_result in epoch_result_list:
            epoch_result[epoch] = {"train_loss": train_loss, "val_loss": val_loss, "val_result":val_result}
        with open(
            os.path.join(self.result_save_dir_path, "epoch_result.json"), "w"
        ) as f:
            json.dump(epoch_result, f, indent=4)
        return epoch_result

    @abstractmethod
    def loss_func(self, y_pred, y_true, *args, **kwargs):
        """
        Abstract method for computing the loss.

        To be implemented by subclasses.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_true
            True values.

        Returns
        -------
        torch.Tensor
            Computed loss.
        """
        pass

    @abstractmethod
    def train_one_epoch(self, *args, **kwargs):
        """
        Abstract method for training the model for one epoch.

        To be implemented by subclasses.
        Returns
        -------
        torch.Tensor
            Train loss.
        """
        pass

    @abstractmethod
    def evaluate(self, data_loader, metrics, *args, **kwargs):
        """
        Abstract method for evaluating the model.

        To be implemented by subclasses.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader for the evaluation data.
        metrics : tuple of str
            Metrics to evaluate the model performance.

        Returns
        -------
        tuple
            loss : torch.Tensor
                Evaluation loss.
            eval_results : list
                List of computed metric values.
            y_pred
                Predicted values.
            y_true
                True values.
        """
        pass

    @staticmethod
    def get_eval_result(y_pred, y_true, metrics=("mae", "rmse", "mape")):
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

            result = eval_func(y_pred, y_true)
            eval_results.append(result)

        return eval_results
