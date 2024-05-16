import torch.nn as nn
import numpy as np
import tqdm
from tqdm import tqdm
import torch
from .abs import AbstractTrainer
#from utils.KL_divergence import KL_divergence


class CrcaaTrainer(AbstractTrainer):
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
            for i, ((s_x, s_y), (t_x, t_y)) in enumerate(zip(s_data_loader, t_data_loader)):
                if len(s_y) != len(t_y):
                    break

                batch_num = s_x.size(0)
                #inputs = torch.cat((s_x, t_x), dim=0)

                domain_label_source = torch.ones(batch_num).float()
                domain_label_target = torch.zeros(batch_num).float()
                domain_label = torch.cat((domain_label_source, domain_label_target), dim=0)

                #inputs = inputs.to(self.device)
                s_x = s_x.to(self.device)
                t_x = t_x.to(self.device)
                s_y = s_y.to(self.device)
                domain_label_source = domain_label_source.to(self.device)
                domain_label_target = domain_label_target.to(self.device)
                domain_label = domain_label.to(self.device)

                #domain_pred, class_pred,feature= self.model(inputs, train=True)
                s_domain_pred, s_class_pred,s_feature= self.model(s_x, train=True)
                t_domain_pred, t_class_pred,t_feature= self.model(t_x, train=True)
                loss_crm_s = 0
                loss_crm_t = 0

                loss_crm_s = self.KL_divergence(s_x,s_feature)+self.KL_divergence(s_feature,s_x)
                loss_crm_t = self.KL_divergence(t_x,t_feature)+self.KL_divergence(t_feature,t_x)

                class_pred = torch.cat((s_class_pred,t_class_pred),dim=0)
                domain_pred = torch.cat((s_domain_pred,t_domain_pred),dim=0)
                #torch.set_printoptions(profile="full",precision=4,sci_mode=False)
                #print(class_pred)
                #print(s_y)
                #torch.set_printoptions(profile="default")
                classification_loss = nn.CrossEntropyLoss()(class_pred.narrow(0, 0, batch_num), s_y.long())
                adversarial_loss = nn.BCELoss()(domain_pred.squeeze(), domain_label)
                #print("classification: "+str(classification_loss))
                #print("adversarial: "+str(adversarial_loss))
                loss = classification_loss + 0.2*adversarial_loss +(loss_crm_s+loss_crm_t)
                #print(str(loss_crm_s)+" "+ str(loss_crm_t))
                
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
            
            self.model.forecast_len = 1

        return self.model
    def KL_divergence(self,P,Q):
        def Gauss_conditional(input):
            batch_size = input.size(0)          
            pji = torch.zeros(batch_size,batch_size)
            pji.to(self.device)        
            P = input.squeeze()
            #print(P)
            cov_P= torch.cov(P.t())
            diff_matrix = (P.unsqueeze(1) - P.unsqueeze(0)).unsqueeze(-1)  # 形状为 [batch_size, batch_size, feature_size]
            tem = torch.matmul(diff_matrix.transpose(-1, -2), cov_P)      
            intermediate_matrix = torch.matmul(tem, diff_matrix)  # 形状为 [batch_size, batch_size, feature_size]
            #print(intermediate_matrix.size())
            pji = intermediate_matrix.squeeze()
            pji.fill_diagonal_(0)
            pji = pji/batch_size
            #print(pji)
            pji = torch.exp(pji * -0.5)  # 形状为 [batch_size, batch_size]
           
            pji = pji / torch.sum(pji, dim=1, keepdim=True)  # 形状为 [batch_size, batch_size]
            
            
        
            
            return pji
        
        pji = Gauss_conditional(P)

        #input("modulepause")
        qji = Gauss_conditional(Q)
        #input("modulepause")
        
        kl_di = torch.sum(pji*torch.log(pji/qji))
        
        
        return 0.05*kl_di



        
        
