for num_epochs in 1; do
    for run_name in norm; do
        for debias_ratio in 0.99; do
            for num_wiki_words in 100; do
                for num_stereo_wiki_words in 10; do
                    python guidebias.py \
                        --num_gender_words 60 \
                        --num_wiki_words ${num_wiki_words} \
                        --num_stereo_wiki_words ${num_stereo_wiki_words} \
                        --model_name_or_path bert-base-uncased \
                        --model_name bert \
                        --output_dir ./out/ \
                        --num_gpus 1 \
                        --batch_size 1024 \
                        --project guidebias \
                        --run_name ${run_name} \
                        --seed 0 \
                        --lr 2e-5 \
                        --num_epochs ${num_epochs} \
                        --num_workers 8 \
                        --grad_accum_steps 1 \
                        --warmup_proportion 0.2 \
                        --debias_ratio ${debias_ratio}
                done
            done
        done
    done
done