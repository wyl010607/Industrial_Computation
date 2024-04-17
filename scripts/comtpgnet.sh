echo "开始测试......"
python ./main.py \
    --data_config_path "./config/data_config/DIST_online_config.yaml" \
    --model_config_path "./config/model_config/OTPGNET_model_config.yaml" \
    --train_config_path "./config/train_config/OTPGNET_train_config.yaml" \
    --model_name "comnet" \
    --model_save_path "./model_states/COMTPGNET/FSNET000.pkl" \
    --result_save_dir "./results/COMTPGNET/FSNET000"
echo "结束测试......"