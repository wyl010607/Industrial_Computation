import warnings

import torch.nn as nn
import tqdm
from tqdm import tqdm
import torch
from .abs import AbstractTrainer


class DasanTrainer(AbstractTrainer):
    """
    A Trainer subclass for iter Fault Diagnosis.
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

    def train_one_epoch(self, s_data_loader, t_data_loader):
        self.model.train()
        total_loss = 0

        with tqdm(total=len(t_data_loader), leave=False) as pbar:
            # batch
            for i, ((s_x, s_y), (t_x, t_y)) in enumerate(
                    zip(s_data_loader, t_data_loader)):

                if len(s_y) != len(t_y):
                    break

                batch_num = s_x.size(0)

                domain_label_source = torch.ones(batch_num).float()
                domain_label_target = torch.zeros(batch_num).float()
                domain_label = torch.cat((domain_label_source, domain_label_target), dim=0)


                # move to GPU if available
                s_x = s_x.to(self.device)
                t_x = t_x.to(self.device)
                s_y = s_y.to(self.device)
                t_y = t_y.to(self.device)
                domain_label = domain_label.to(self.device)

                domain_pred, class_pred, lmmd_loss, pseudo_loss = self.model(t_x, s_x, s_y, train=True)
                pred = torch.max(class_pred.cpu(), 1)[1].numpy()
                d_pred = torch.max(domain_pred.cpu(), 1)[1].numpy()

                classification_loss = nn.CrossEntropyLoss()(class_pred.narrow(0, 0, batch_num),
                                                            s_y.long())
                adversarial_loss = nn.BCELoss()(domain_pred.squeeze(), domain_label)
                loss = classification_loss + adversarial_loss + lmmd_loss + pseudo_loss

                # print("inputs:\n{}".format(inputs))
                # # print("domain_pred:\n{}".format(domain_pred))
                # print("class_pred:\n{}".format(class_pred))
                #
                # print("class_pred:\n{}".format(pred))
                # print("domain_pred:\n{}".format(d_pred))
                # print("loss:\n{}".format(loss))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description("loss is {:.4f}".format(loss.item()))
                pbar.update()
                total_loss += loss.item()

        return total_loss / len(t_data_loader)

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

        y_true, y_pred, tol_loss, data_num, correct_num = [], [], 0, 0, 0
        for i, (x, y) in enumerate(data_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            batch_num, total_num = x.size(0), 0
            class_pred = self.model(x, train=False)

            pred = torch.max(class_pred.cpu(), 1)[1].numpy()
            correct_num += (pred == y.cpu().numpy()).sum()
            total_num += len(y)
            loss = nn.CrossEntropyLoss()(class_pred.narrow(0, 0, batch_num), y.long()).cpu().numpy()

            tol_loss += loss
            data_num += 1
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred)
        eval_results = self.get_eval_result(y_true, y_pred, metrics)
        print("Evaluate result: ", end=" ")
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")


        return (
            tol_loss / data_num,
            zip(metrics, eval_results),
            y_pred,
            y_true,
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
