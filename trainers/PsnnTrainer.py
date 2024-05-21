import warnings

import torch.nn as nn
import tqdm
from tqdm import tqdm
import torch
import itertools
import torch.optim as optim
from .abs import AbstractTrainer
import numpy as np


class PsnnTrainer(AbstractTrainer):
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
        learning_rate = 0.0005
        optimizer_c1 = optim.SGD(self.model.classifier1.parameters(), lr=learning_rate)
        optimizer_c2 = optim.SGD(self.model.classifier2.parameters(), lr=learning_rate)
        optimizer_g = optim.SGD(self.model.feature_extractor.parameters(), lr=learning_rate)
        optimizerd = optim.SGD(self.model.domain_discriminator.parameters(), lr=learning_rate)

        with tqdm(total=len(t_data_loader), leave=False) as pbar:
            # batch
            for i, ((s_x, s_y), (t_x, t_y)) in enumerate(zip(s_data_loader, t_data_loader)):
                if len(s_y) != len(t_y):
                    break

                batch_num = s_x.size(0)
                inputs = torch.cat((s_x, t_x), dim=0)

                domain_label_source = torch.ones(batch_num).float()
                domain_label_target = torch.zeros(batch_num).float()
                domain_label = torch.cat((domain_label_source, domain_label_target), dim=0)

                inputs = inputs.to(self.device)
                s_y = s_y.to(self.device)
                domain_label = domain_label.to(self.device)

                domain_pred, class_pred1,class_pred2 = self.model(inputs, train=True)

                classification_loss1 = nn.CrossEntropyLoss()(class_pred1.narrow(0, 0, batch_num), s_y.long())
                
                classification_loss2 = nn.CrossEntropyLoss()(class_pred2.narrow(0, 0, batch_num), s_y.long())
                adversarial_loss = nn.BCELoss()(domain_pred.squeeze(), domain_label)

                loss_s = classification_loss1+classification_loss2
                optimizer_g.zero_grad()
                optimizer_c1.zero_grad()
                optimizer_c2.zero_grad()
                loss_s.backward()
                optimizer_g.step()
                optimizer_c1.step()
                optimizer_c2.step()

                
                
                
                

                domain_pred, class_pred1,class_pred2 = self.model(inputs, train=True)

                #CDD =  torch.sum(torch.matmul(class_pred1.unsqueeze(1), class_pred1.unsqueeze(2))-torch.matmul(class_pred1.unsqueeze(1), class_pred2.unsqueeze(2)))   
                CDD = torch.dist(class_pred1,class_pred2,p=1)*0.5
                
                CDD = CDD/batch_num/1000
                lc = nn.CrossEntropyLoss()(class_pred1.narrow(0, 0, batch_num), s_y.long())+nn.CrossEntropyLoss()(class_pred2.narrow(0, 0, batch_num), s_y.long())
                divergence_loss = lc-CDD
                optimizer_c1.zero_grad()
                optimizer_c2.zero_grad()

                divergence_loss.backward()
                optimizer_c1.step()
                optimizer_c2.step()

                
                
                
                domain_pred, class_pred1,class_pred2 = self.model(inputs, train=True)
                #CDD =  torch.sum(torch.matmul(class_pred1.unsqueeze(1), class_pred1.unsqueeze(2))-torch.matmul(class_pred1.unsqueeze(1), class_pred2.unsqueeze(2)))              
                CDD = torch.dist(class_pred1,class_pred2,p=1)*0.5
                CDD = CDD/batch_num/1000
                lc = nn.CrossEntropyLoss()(class_pred1.narrow(0, 0, batch_num), s_y.long())+nn.CrossEntropyLoss()(class_pred2.narrow(0, 0, batch_num), s_y.long())
                #print(CDD)
                divergence_loss1 = CDD +lc
                divergence_loss2 = CDD +lc
                optimizer_g.zero_grad()
                
                divergence_loss1.backward(retain_graph = True)
                divergence_loss2.backward()
                optimizer_g.step()
                



                loss = (classification_loss1+classification_loss2)/2
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
