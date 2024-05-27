import warnings
from torch.nn import functional as F
import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer
import torch.nn as nn


class MLPTrainer(AbstractTrainer):
    """
    A Trainer subclass for iter multi-step forecasting.
    """

    def __init__(
        self,
        model,
        optimizer,
        scaler,
        scheduler,
        model_save_path,
        result_save_dir_path,
        max_epoch_num,
        enable_early_stop=True,
        early_stop_patience=50,
        early_stop_min_is_best=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            model,
            optimizer,
            scaler,
            scheduler,
            model_save_path,
            result_save_dir_path,
            max_epoch_num,
            enable_early_stop,
            early_stop_patience,
            early_stop_min_is_best,
            *args,
            **kwargs,
        )
        # self._check_model_is_single_step()

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        loss = F.mse_loss(y_pred, y_true)
        return loss

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        

        tqmd_ = tqdm(data_loader)
        for data in tqmd_:
            x=data[0]
            y=data[1]
            x = x.type(torch.float).to(
                self.device
            )  
            y = y.type(torch.float).to(self.device).unsqueeze(1)
            prediction = self.model(x)
            loss = self.loss_func(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tqmd_.set_description("loss is {:.4f}".format(loss.item()))
            total_loss += loss.item()
        return total_loss / len(data_loader)

    @torch.no_grad()
    def evaluate(self, data_loader,test,metrics, **kwargs):
        
        self.model.eval()

        y_true, y_pred, tol_loss, data_num = [], [], 0, 0
        for data in data_loader:
            x=data[0]
            y=data[1]
            x = x.type(torch.float).to(self.device)
            if test:
              x = x.type(torch.float).to(self.device).squeeze(1)
              
            y = y.type(torch.float).to(self.device).unsqueeze(1)
            prediction = self.model(x)
            loss = self.loss_func(prediction, y).item()
            tol_loss += loss
            data_num += 1
            # for i in range(y.size(0)):
            #     y[i]=y[i]*(max-min)+min
            #     # y[i]=torch.round(y[i])
            # for i in range(prediction.size(0)):
            #     prediction[i]=prediction[i]*(max-min)+min
            #     # prediction[i]=torch.round(prediction[i])
            y_true.append(y)
            y_pred.append(prediction)

        # loss_L1 = nn.L1Loss()
        
      
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        y_true = torch.cat(y_true, dim=0).cpu().numpy()

        eval_results = self.get_eval_result(y_pred, y_true, metrics)
        print("Evaluate result: ", end=" ")
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
        print()
        # reshape y_pred to [batch_size * len(data_loader), time_step, feature_size]
        return (
            tol_loss / data_num,
            zip(metrics, eval_results),
            y_pred,
            y_true,
        )

