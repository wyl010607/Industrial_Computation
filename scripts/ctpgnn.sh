echo "开始测试......"
python ./main.py \
    --data_config_path "./config/data_config/DIST_config.yaml" \
    --model_config_path "./config/model_config/CTPGNN_model_config.yaml" \
    --train_config_path "./config/train_config/CTPGNN_train_config.yaml" \
    --model_name "CSTAGNN_stamp" \
    --model_save_path "./model_states/CTPGNN/CTPGNN3.pkl" \
    --result_save_dir "./results/CTPGNN3"
echo "结束测试......"