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
from torch.utils.data import DataLoader, TensorDataset


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
    random_seed = train_config["random_seed"]
    device = torch.device(train_config["device"])
    batch_size = train_config["batch_size"]
    load_checkpoint = train_config["load_checkpoint"]

    # set random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ----------------------,- Load data ------------------------
    data_preprocessor_class = getattr(
        sys.modules["data_preprocessors"], data_config["data_preprocessor_name"]
    )
    data_preprocessor = data_preprocessor_class(**data_config["data_preprocessor_params"])
    preprocessed_data = data_preprocessor.preprocess()
    s_train_data, s_train_label, t_train_data, t_train_label, t_val_data, t_val_label, t_test_data, t_test_label = data_preprocessor.split_data(
        preprocessed_data)

    # update model & trainer params
    model_config[args.model_name].update(data_preprocessor.update_model_params)
    train_config["trainer_params"].update(data_preprocessor.update_trainer_params)

    # scale data
    scaler_class = getattr(sys.modules["utils.scaler"], data_config["scaler_name"])
    scaler = scaler_class(**data_config["scaler_params"])
    scaler.fit(s_train_data)
    s_train_data = scaler.transform(s_train_data)
    scaler.fit(t_train_data)
    t_train_data = scaler.transform(t_train_data)
    scaler.fit(t_val_data)
    t_val_data = scaler.transform(t_val_data)
    scaler.fit(t_test_data)
    t_test_data = scaler.transform(t_test_data)

    # 将其他几个变量从 NumPy 数组转换为 PyTorch 张量
    s_train_data = torch.from_numpy(s_train_data)
    s_train_label = torch.from_numpy(s_train_label)
    t_train_data = torch.from_numpy(t_train_data)
    t_train_label = torch.from_numpy(t_train_label)
    t_val_data = torch.from_numpy(t_val_data)
    t_val_label = torch.from_numpy(t_val_label)
    t_test_data = torch.from_numpy(t_test_data)
    t_test_label = torch.from_numpy(t_test_label)

    s_train_dataset = TensorDataset(s_train_data, s_train_label)
    t_train_dataset = TensorDataset(t_train_data, t_train_label)
    t_val_dataset = TensorDataset(t_val_data, t_val_label)
    t_test_dataset = TensorDataset(t_test_data, t_test_label)

    s_train_dataloader = DataLoader(
        s_train_dataset, batch_size=batch_size, **data_config["dataloader_params"]
    )
    t_train_dataloader = DataLoader(
        t_train_dataset, batch_size=batch_size, **data_config["dataloader_params"]
    )
    valid_dataloader = DataLoader(
        t_val_dataset, batch_size=batch_size, **data_config["dataloader_params"]
    )
    test_dataloader = DataLoader(
        t_test_dataset, batch_size=batch_size, **data_config["dataloader_params"]
    )

    # ------------------------- Model ---------------------------

    # model
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

    epoch_results = trainer.train(s_train_dataloader, t_train_dataloader, valid_dataloader)
    test_result, y_pred, y_true = trainer.test(test_dataloader)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="./config/train_config/DANN_train_config.yaml",
        help="Config path of Trainer",
    )

    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./config/model_config/DANN_model_config.yaml",
        help="Config path of models",
    )

    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/data_config/CWRU_config.yaml",
        help="Config path of Data",
    )
    parser.add_argument("--model_name", type=str, default="DANN", help="Model name")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./model_states/DANN_PU.pkl",
        help="Model save path",
    )

    parser.add_argument(
        "--result_save_dir_path",
        type=str,
        default="./results/DANN_PU",
        help="Result save path",
    )
    args = parser.parse_args()
    main(args)
