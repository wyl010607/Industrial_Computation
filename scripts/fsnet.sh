echo "开始测试......"
python ./main.py \
    --data_config_path "./config/data_config/ETTh2_config.yaml" \
    --model_config_path "./config/model_config/FSNET_model_config.yaml" \
    --train_config_path "./config/train_config/FSNET_train_config.yaml" \
    --model_name "net" \
    --model_save_path "./model_states/FSNET/FSNET2.pkl" \
    --result_save_dir "./results/FSNET2"
echo "结束测试......"