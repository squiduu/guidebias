python run_crows_pairs.py \
    --test_data_path ./data/crows_pairs_anonymized.csv \
    --bias_type gender \
    --model_name_or_path bert-base-uncased \
    --model_name bert-base-uncased \
    --output_dir ./out/ \
    --run_name run00 \
    --seed 0