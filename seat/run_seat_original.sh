python run_seat.py \
    --tests seat6,seat6b,seat7,seat7b,seat8,seat8b \
    --model_name bert \
    --output_dir ./out/ \
    --enc_save_dir ../data/tests/ \
    --cache_encs \
    --data_dir ../data/tests/ \
    --num_samples 100000 \
    --run_name run00 \
    --version bert-base-uncased