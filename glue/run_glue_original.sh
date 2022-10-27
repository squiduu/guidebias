for task_name in cola mnli mrpc qnli qqp rte sst2 stsb wnli; do
    for seed in 0 1 2; do
        python run_glue.py \
            --model_name_or_path bert-base-uncased \
            --task_name ${task_name} \
            --seed ${seed} \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --output_dir ./out/run00/${seed}/${task_name} \
            --run_name run00 \
            --overwrite_output_dir \
            --save_strategy no \
            --logging_strategy no \
            --fp16
    done
done