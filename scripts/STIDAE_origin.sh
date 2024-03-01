echo "开始测试......"
python ./main.py \
    --data_config_path ./config/data_config/DIST_with_pretrain_config.yaml \
    --model_config_path ./config/model_config/pretrain_model_config.ymal \
    --train_config_path ./config/train_config/train_with_STIDAE_pretrain_config.yaml \
    --model_name STIDAE \
    --model_save_path ./model_states/jl_STIDAE_origin.pkl \
    --result_save_dir ./results/jl_STIDAE_origin \

echo "结束测试......"
