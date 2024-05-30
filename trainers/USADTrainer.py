from tqdm import tqdm
import numpy as np
import torch
from models.USAD.usad import USADModel
from trainers.abs import AbstractTrainer
from data_preprocessors.USADSDataPreprocessor import label_len
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report,roc_auc_score
from utils.metrics import ROC


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


device = get_default_device()


class USADTrainer(AbstractTrainer):
    """
    A Trainer subclass for Model USAD.
    """

    def __init__(
        self,
        model,
        optimizer1,
        optimizer2,
        scheduler,
        scaler,
        model_save_path,
        result_save_dir_path,
        max_epoch_num,
        batch=None,
        enable_early_stop=False,
        early_stop_patience=5,
        early_stop_min_is_best=True,
        alpha=0.1,
        beta=0.9,
        window_size=10,
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
            optimizer1,
            optimizer2,
            scheduler,
            scaler,
            model_save_path,
            result_save_dir_path,
            max_epoch_num,
            batch,
            enable_early_stop,
            early_stop_patience,
            early_stop_min_is_best,
            alpha,
            beta,
            window_size
            *args,
            **kwargs,
        )
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size

    def loss_func(self, batch, epoch, *args, **kwargs):
        loss1, loss2 = self.model.training_step(batch,epoch+1)
        return loss1, loss2

    def val_loss(self, val_loader, n):
        outputs = [self.model.validation_step(to_device(batch, device), n) for [batch] in val_loader]
        return self.model.validation_epoch_end(outputs)

    def train_one_epoch(self, train_loader, val_loader, epoch, *args, **kwargs):
        self.model.train()
        # total_loss = 0
        # tqmd = tqdm(data_loader)
        for [batch] in train_loader:
            batch = to_device(batch, device)
            # Train AE1
            loss1, loss2 = self.loss_func(batch, epoch + 1)
            loss1.backward()
            self.optimizer1.step()
            self.optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = self.model.training_step(batch, epoch + 1)
            loss2.backward()
            self.optimizer2.step()
            self.optimizer2.zero_grad()
        # 返回验证损失
        val_result = self.val_loss(self.model, val_loader, epoch + 1)
        # val_result = self.val_loss(self.model, train_loader, epoch+1)
        return val_result

    def evaluate(self, data_loader, metrics, save_graph=False, *args, **kwargs):
        self.model.eval()
        test_results = []
        me_list=[]

        with torch.no_grad():
            for [batch] in data_loader:
                batch = to_device(batch, device)
                w1 = self.model.decoder1(self.model.encoder(batch))
                w2 = self.model.decoder2(self.model.encoder(w1))
                test_results.append(
                    self.alpha * torch.mean((batch - w1) ** 2, axis=1) + self.beta * torch.mean((batch - w2) ** 2, axis=1))
        # 简化版point-adjust技术，Anomaly-Transformer是应用该技术最重要的方法之一，因此后两个模型有该技术的完整版实现
        windows_labels = []
        labels, len_l = label_len()
        for i in range(len_l - self.window_size):
            windows_labels.append(list(np.int_(labels[i:i + self.window_size])))
        y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
        y_pred = np.concatenate([torch.stack(test_results[:-1]).flatten().detach().cpu().numpy(),
                                 test_results[-1].flatten().detach().cpu().numpy()])
        # 异常检测
        threshold = 0.85
        '''
        网格搜索
        thresholds = np.arange(0.0, np.max(y_pred), np.max(y_pred) / 50)
        fscore = np.zeros(shape=(len(thresholds)))
        for index, elem in enumerate(thresholds):
            y_pred_prob = (y_pred > elem).astype('int')
            fscore[index] = f1_score(y_test, y_pred_prob)
        index = np.argmax(fscore)
        thresholdOpt = round(thresholds[index], ndigits=4)
        fscoreOpt = round(fscore[index], ndigits=4)
        y_pred_prob = (y_pred > thresholds[index]).astype('int')
        acc = accuracy_score(y_test, y_pred_prob)
        recall = recall_score(y_test, y_pred_prob)
        '''
        y_pred_label = [1.0 if (score > threshold) else 0 for score in y_pred]
        prec = precision_score(y_test, y_pred_label, pos_label=1)
        recall = recall_score(y_test, y_pred_label, pos_label=1)
        f1 = f1_score(y_test, y_pred_label, pos_label=1)
        auc = roc_auc_score(y_test, y_pred_label)
        print(prec, recall, f1)

        me_list = [f1,prec,recall,auc]
        #test_loss = self.val_loss(self.model, data_loader, epoch + 1)
        # 返回计算得到的异常评分列表
        return test_results, zip(metrics, me_list), y_pred, y_test
