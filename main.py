import json
import os
import sys
import random
import yaml
import argparse
import numpy as np
import torch

import trainers
import models
import datasets
import data_preprocessors

from utils import scaler
from torch.utils.data import DataLoader


def load_config(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    result_save_dir_path = args.result_save_dir_path
    model_save_path = args.model_save_path
    if not os.path.exists(result_save_dir_path):
        os.makedirs(result_save_dir_path)
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
   
    train_config = load_config(args.train_config_path)
    model_config = load_config(args.model_config_path)
    data_config = load_config(args.data_config_path)

    # basic config
    
    # random_seed = train_config["random_seed"]
    device = torch.device(train_config["device"])
    batch_size = train_config["batch_size"]
    load_checkpoint = train_config["load_checkpoint"]

    # set random seeds for reproducibility

    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # np.random.seed(random_seed)
    # random.seed(random_seed)

    # ----------------------- Load data ------------------------
    
    
    train_dataloader, test_dataloader, valid_dataloader = getattr(sys.modules["datasets"], data_config["dataset_name"]).get_data_loaders(
        data_path=data_config['dataset_params']['data_path'],
        sequence_len=data_config['dataset_params']['sequence_len'],
        sub_dataset=data_config['dataset_params']['sub_dataset'],
        norm_type=data_config['dataset_params']['norm_type'],
        max_rul=data_config['dataset_params']['max_rul'],
        cluster_operations=data_config['dataset_params']['cluster_operations'],
        norm_by_operations=data_config['dataset_params']['norm_by_operations'],
        use_max_rul_on_test=data_config['dataset_params']['use_max_rul_on_test'],
        validation_rate=data_config['dataset_params']['validation_rate'],
        return_id=True,
        use_only_final_on_test=data_config['dataset_params']['use_only_final_on_test'],
        loader_kwargs={'batch_size':data_config['dataset_params']['batch_size']},
        emd=data_config['dataset_params']['emd'],
        emd_num=data_config['dataset_params']['emd_num']
    )

    # train_dataloader, test_dataloader, valid_dataloader = getattr(sys.modules["datasets"], data_config["dataset_name"]).get_data_loaders(**data_config["dataset_params"])

    # train_dataloader, test_dataloader, valid_dataloader,minmax_dict = getattr(sys.modules["datasets"], data_config["dataset_name"]).getDataloader(
    #   data_path=data_config['dataset_params']['data_path'],
    #   sub_dataset=data_config['dataset_params']['sub_dataset']
    # )
   





    #data_preprocessor_class = getattr(
        #sys.modules["data_preprocessors"], data_config["data_preprocessor_name"]
    #)
    #dcata_preproessor = data_preprocessor_class(**data_config["data_preprocessor_params"])
    #df_train,df_valid,df_test,train_indices,val_indices,units= data_preprocessor.preprocess()
    # dataset
    # datasets_class = getattr(sys.modules["datasets"], data_config["dataset_name"])
    # datasets = datasets_class(**data_config["dataset_params"])
    #train,val,test=datasets_class(train_indices, valid_indices, df_train,df_valid,units,df_test)
    # train_dataloader = DataLoader(
    #     train, batch_size=64, shuffle=True)
    # valid_dataloader = DataLoader(
    #     val, batch_size=len(valid_indices), shuffle=True)
    # test_dataloader = DataLoader(
    #     test, batch_size=100, shuffle=True)

    # update model & trainer params
    # data_config["dataset_params"].update(data_preprocessor.update_dataset_params)
    # model_config[args.model_name].update(data_preprocessor.update_model_params)
    # train_config["trainer_params"].update(data_preprocessor.update_trainer_params)

    # scale data
    # scaler_class = getattr(sys.modules["utils.scaler"], data_config["scaler_name"])
    # scaler = scaler_class(**data_config["scaler_params"])
    # scaler.fit(train_data)
    # train_data = scaler.transform(train_data)
    # valid_data = scaler.transform(valid_data)
    # test_data = scaler.transform(test_data)

    
    # ------------------------- Model ---------------------------

    # model2
    model_class = getattr(sys.modules["models"], args.model_name)
    model = model_class(**model_config[args.model_name])
    model.to(device)
    # ------------------------- Trainer -------------------------

    # Optimizer
    optimizer_class = getattr(
        sys.modules["torch.optim"], train_config["optimizer_name"]
    )
    optimizer = optimizer_class(model.parameters(), **train_config["optimizer_params"])

    # scheduler
    scheduler_class = getattr(
        sys.modules["torch.optim.lr_scheduler"], train_config["scheduler_name"]
    )
    scheduler = scheduler_class(optimizer, **train_config["scheduler_params"])

    print(model_save_path)
    # trainer
    trainer_class = getattr(sys.modules["trainers"], train_config["trainer_name"])

    trainer = trainer_class(
        model,
        optimizer,
        scheduler,
        scaler,
        model_save_path,
        result_save_dir_path,
        **train_config["trainer_params"]
    )

    # load checkpoint
    if load_checkpoint:
        trainer.load_checkpoint()
 
    # ------------------------- Train & Test ------------------------
    config = {
        "args": vars(args),
        "train_config": train_config,
        "model_config": model_config,
        "data_config.yaml": data_config,
    }
    print("Configuration: ", config)
    with open(os.path.join(result_save_dir_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("Start training.")

    # epoch_results = trainer.train(train_dataloader, valid_dataloader,minmax_dict['rul'+'min'],minmax_dict['rul'+'max'])
    # test_result, y_pred, y_true = trainer.test(test_dataloader,minmax_dict['rul1'+'min'],minmax_dict['rul1'+'max'])

    epoch_results = trainer.train(train_dataloader, valid_dataloader)
    test_result, y_pred, y_true = trainer.test(test_dataloader)

    # save y_pred, y_true to self.result_save_dir/y_pred.npy, y_true.npy
    np.save(os.path.join(result_save_dir_path, "test_y_pred.npy"), y_pred)
    np.save(os.path.join(result_save_dir_path, "test_y_true.npy"), y_true)

    # save results
    result = {
        "config": config,
        "test_result": test_result,
        "epoch_results": epoch_results,
    }
    with open(os.path.join(result_save_dir_path, "result.json"), "w") as f:
        json.dump(result, f, indent=4)

    print("Training finished.")


if __name__ == "__main__":
    # data1 = np.load("./results/LSTM_FD003/test_y_pred.npy")
    # data2 = np.load("./results/LSTM_FD003/test_y_true.npy")
    # print(data1)
    # print(" ")
    # print(data2)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="./config/train_config/MA_train_config.yaml",
        help="Config path of Trainer",
    )

    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./config/model_config/MA_model_config.yaml",
        help="Config path of models",
    )

    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/data_config/MA_config.yaml",
        help="Config path of Data",
    )
    parser.add_argument("--model_name", type=str, default="ModeAttention", help="Model name")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./model_states/MA.pkl",
        help="Model save path",
    )

    parser.add_argument(
        "--result_save_dir_path",
        type=str,
        default="./results/MA",
        help="Result save path",
    )
    args = parser.parse_args()
    main(args)
