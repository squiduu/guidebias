python run_stereoset.py \
    --test_data_path ./data/test.json \
    --bias_type gender \
    --model_name_or_path bert-base-uncased \
    --output_dir ./out \
    --run_name run00 \
    --seed 0 \
    --per_device_batch_size 1