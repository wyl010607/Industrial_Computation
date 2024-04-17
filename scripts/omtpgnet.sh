echo "开始测试......"
python ./main.py \
    --data_config_path "./config/data_config/PEMS-BAY_config.yaml" \
    --model_config_path "./config/model_config/OTPGNET_model_config.yaml" \
    --train_config_path "./config/train_config/OTPGNET_train_config.yaml" \
    --model_name "omnet" \
    --model_save_path "./model_states/OMTPGNET/FSNET010.pkl" \
    --result_save_dir "./results/OMTPGNET/FSNET010"
echo "结束测试......"