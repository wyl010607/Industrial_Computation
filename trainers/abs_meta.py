import json
import os
import time
from shutil import copyfile
import torch
from abc import ABC, abstractmethod

from utils.early_stop import EarlyStop, EarlyStopping
import utils.metrics as metrics_module


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


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
        optimizer,
        scaler,
        model_save_path,
        result_save_dir_path,
        max_epoch_num,
        dataset,
        lr,
        # enable_early_stop=False,
        early_stop_patience=5,
        # early_stop_min_is_best=True,
        *args,
        **kwargs,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.model_save_path = model_save_path
        self.model_save_dir_path = os.path.dirname(model_save_path)
        self.result_save_dir_path = result_save_dir_path
        self.epoch_now = 0
        self.max_epoch_num = max_epoch_num
        self.early_stop_patience = early_stop_patience
        self.dataset = dataset
        self.lr = lr
        early_stopping = EarlyStopping(self.early_stop_patience, verbose=True, dataset_name=self.dataset)
        self.early_stopping = early_stopping
        # self.enable_early_stop = enable_early_stop
        # if self.enable_early_stop:
            # self.early_stop = EarlyStop(early_stop_patience, early_stop_min_is_best)
        # else:
            # self.early_stop = None
        self.device = next(self.model.parameters()).device
        self.min_loss = torch.finfo(torch.float32).max

    def train(
        self,
        train_data_loader,
        test_data_loader,
        thre_data_loader,
        support,
        query,
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
            train_steps = len(train_data_loader)
            train_loss = self.train_one_epoch(train_data_loader, test_data_loader, self.epoch_now, time_now, train_steps,support, query)  # 训练一个epoch
            self.epoch_now += 1
            # print(f"Train loss: {train_loss:.4f}")
            # evaluate
            # eval_loss, metrics_evals, _, _ = self.evaluate(thre_data_loader, metrics)
            print("模型验证：")
            eval_loss, prec, recall, f1, auc, _, _ = self.evaluate(train_data_loader, thre_data_loader, metrics, self.epoch_now)
            if epoch==2:
                epoch_result_list.append([epoch, train_loss, eval_loss, prec,recall, f1, auc])
            # print("epoch类型：",type(epoch),",train_loss类型：",type(train_loss),",eval_loss类型：",type(eval_loss),",prec类型：",type(prec),"recall类型：",type(recall),"，f1类型：",type(f1),",auc类型：",type(auc))

            if self.early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
            # check early stop
            # if self.early_stop is not None and self.early_stop.reach_stop_criteria(
                # eval_loss
            #):
                # self.early_stop.reset()
                # break

            # save best model
            # if eval_loss < self.min_loss:
                # self.min_loss = eval_loss
            torch.save(self.model.state_dict(), tmp_state_save_path)

            # lr scheduler step
            # if self.scheduler is not None:
                # self.scheduler.step()
        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
        epoch_result_json = self._save_epoch_result(epoch_result_list)  # 保存epoch结果
        return epoch_result_json

    def test(self, train_data_loader, test_data_loader, metrics=("prec", "rec", "f1", "auc"), *args, **kwargs):
        # load best model
        self.model.load_state_dict(torch.load(self.model_save_path))
        # evaluate on test set
        print("模型测试：")
        self.epoch_now = self.epoch_now+1
        test_loss, prec, recall, f1, auc, y_pred, y_true = self.evaluate(train_data_loader, test_data_loader, metrics, self.epoch_now)
        test_result = {"loss": test_loss, "precision": prec, "recall":recall, "f1_score":f1, "auc_score":auc}
        # for metric_name, metric_eval in metrics_evals:
            # test_result[metric_name] = metric_eval
        return test_result, y_pred, y_true

    def save_checkpoint(self, filename="checkpoint.pth"):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # "scheduler_state_dict": self.scheduler.state_dict()
            # if self.scheduler
            # else None,
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
        for epoch, train_loss, eval_loss, prec,recall, f1, auc in epoch_result_list:
            epoch_result[epoch] = {"train_loss": train_loss, "eval_loss": eval_loss, "precision": prec, "recall":recall, "f1_score":f1, "auc_score":auc}
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
