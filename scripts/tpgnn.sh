echo "开始测试......"
python ./main.py \
    --data_config_path "./config/data_config/PEMSD7_config.yaml" \
    --model_config_path "./config/model_config/TPGNN_model_config.yaml" \
    --train_config_path "./config/train_config/TPGNN_train_config.yaml" \
    --model_name "STAGNN_stamp" \
    --model_save_path "./model_states/TPGNN/TPGNN2.pkl" \
    --result_save_dir "./results/TPGNN2"
echo "结束测试......"