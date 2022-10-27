python run_crows_pairs.py \
    --test_data_path ./data/crows_pairs_anonymized.csv \
    --bias_type gender \
    --model_name_or_path ../guidebias/out/bert_guidebias_ep1_seed0_num100-10 \
    --model_name bert-base-uncased \
    --output_dir ./out/ \
    --run_name run00 \
    --seed 0