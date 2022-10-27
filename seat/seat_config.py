import argparse

TEST_EXT = ".jsonl"


def get_seat_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tests",
        type=str,
        help=f"WEAT tests to run (a comma-separated list; test files should be in `data_dir` and have corresponding \
            names, with extension {TEST_EXT}). Default: all tests.",
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str, default="./out/", help="A directory to log to.")
    parser.add_argument("--cache_encs", action="store_true", help="If set, do not cache encodings to disk.")
    parser.add_argument("--data_dir", type=str, help="Directory containing examples for each test")
    parser.add_argument("--debiased", action="store_true", help="Whether or not to use debiased model.")
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of permutation test samples used when estimate p-values (exact test is used if there are \
            fewer than this many permutations)",
        default=100000,
    )
    parser.add_argument(
        "--use_parametric", action="store_true", help="Use parametric test (normal assumption) to compute p-values."
    )
    parser.add_argument("--run_name", type=str, help="A run number for recording.")
    parser.add_argument("--use_ckpt", action="store_true", help="Whether or not to use checkpoint.")
    parser.add_argument("--ckpt_dir", type=str, help="A directory of checkpoint containing config.json file.")
    parser.add_argument("--version", type=str, help="A model version from HuggingFace.")
    parser.add_argument("--deterministic", action="store_true", help="Whether or not to use checkpoint.")
    parser.add_argument("--enc_save_dir", type=str, help="Whether or not to use checkpoint.")

    return parser.parse_args()
