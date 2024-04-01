echo "开始测试......"
python ./main.py \
    --data_config_path "./config/data_config/DIST_online_config.yaml" \
    --model_config_path "./config/model_config/CFSNET_model_config.yaml" \
    --train_config_path "./config/train_config/CFSNET_train_config.yaml" \
    --model_name "cnet" \
    --model_save_path "./model_states/CFSNET/CFSNET4.pkl" \
    --result_save_dir "./results/CFSNET4"
echo "结束测试......"