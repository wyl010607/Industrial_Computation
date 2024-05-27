import warnings
from torch.nn import functional as F
import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer
import torch.nn as nn
import os
from shutil import copyfile


class LSTMTrainer(AbstractTrainer):
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



    def train(
        self,
        train_data_loader,
        eval_data_loader,
        min,
        max,
        metrics=("rmse","mae"),
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

        for epoch in range(self.epoch_now, self.max_epoch_num):
            print(f"Epoch {epoch} / {self.max_epoch_num}")
            self.save_checkpoint()
            # train
            train_loss = self.train_one_epoch(train_data_loader)  # 训练一个epoch
            self.epoch_now += 1
            print(f"Train loss: {train_loss:.4f}")
            # evaluate
            eval_loss, metrics_evals, _, _ = self.evaluate(eval_data_loader,False, min,max,metrics)
            print(f"eval loss: {eval_loss:.4f}")
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


    def test(self, test_data_loader,min,max, metrics=( "rmse","mae"), *args, **kwargs):
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
        test_loss, metrics_evals, y_pred, y_true = self.evaluate(
            test_data_loader,True ,min,max,metrics
        )
        
        # plt.plot(np.arange(1,len(y_true)+1), y_true, label= 'TrueRUL')
        # plt.plot(np.arange(1,len(y_pred)+1), y_pred, label = 'PedictionRUL')
        # plt.legend()
        # plt.grid()
        # plt.xlabel('NO.')
        # plt.ylabel('RUL')
        # plt.ion()
        # plt.pause(600000)
        # plt.close()

        test_result = {"loss": test_loss}
        for metric_name, metric_eval in metrics_evals:
            test_result[metric_name] = metric_eval
        return test_result, y_pred, y_true

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
            )  # [batch_size, history_len, num_vars, channels]
            # print(x.shape)
            # exit(0)
            y = y.type(torch.float).to(self.device)#.unsqueeze(1)
            prediction = self.model(x)
            loss = self.loss_func(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tqmd_.set_description("loss is {:.4f}".format(loss.item()))
            total_loss += loss.item()
        return total_loss / len(data_loader)

    @torch.no_grad()
    def evaluate(self, data_loader,test,min,max,metrics, **kwargs):
        
        self.model.eval()

        y_true, y_pred, tol_loss, data_num = [], [], 0, 0
        for data in data_loader:
            x=data[0]
            y=data[1]
            x = x.type(torch.float).to(self.device)
            if test:
              x = x.type(torch.float).to(self.device)
              
            y = y.type(torch.float).to(self.device)#.unsqueeze(1)
            prediction = self.model(x)
            loss = self.loss_func(prediction, y).item()
            tol_loss += loss
            data_num += 1
            for i in range(y.size(0)):
                y[i]=y[i]*(max-min)+min
                # y[i]=torch.round(y[i])
            for i in range(prediction.size(0)):
                prediction[i]=prediction[i]*(max-min)+min
                # prediction[i]=torch.round(prediction[i])
            y_true.append(y)
            y_pred.append(prediction)

        # loss_L1 = nn.L1Loss()
        
      
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        y_true = torch.cat(y_true, dim=0).cpu().numpy()

        # test_loss_MSE = ((y_pred - y_true) ** 2).mean() #loss_fn(y_pred, y).item()
        # test_loss_L1 = loss_L1(torch.from_numpy(y_pred),torch.from_numpy(y_true)).item()
        # test_ASUE = ((torch.relu(torch.from_numpy(y_true)-torch.from_numpy(y_pred))).mean()).item()

        # print(f"test_loss_MSE: {test_loss_MSE:.4f}")
        # print(f"test_loss_L1: {test_loss_L1:.4f}")
        # print(f"test_ASUE: {test_ASUE:.4f}")

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

    # def _check_model_is_single_step(self):
    #     """
    #     Check if the model is a single step forecasting model. If not, try to set the forecast_len to 1.
    #     """

    #     if not hasattr(self.model, "forecast_len"):
    #         raise AttributeError(
    #             "The model does not have the attribute 'forecast_len'."
    #         )
    #     if self.model.forecast_len != 1:
    #         warnings.warn(
    #             "The forecast_len of the model is not 1, it will be set to 1."
    #         )
    #         self.model.forecast_len = 1

    #     return self.model
