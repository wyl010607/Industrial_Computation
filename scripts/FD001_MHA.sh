echo "开始测试......"
python ./main.py \
    --data_config_path ./config/data_config/CMAPSS_config.yaml \
    --model_config_path ./config/model_config/MHA_model_config.ymal \
    --train_config_path ./config/train_config/MHA_train_config.yaml \
    --model_name MultiHeadAttentionLSTM \
    --model_save_path ./model_states/MHA.pkl \
    --result_save_dir ./result/MHA \
