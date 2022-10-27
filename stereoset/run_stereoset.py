import torch
import os
import json
from logging import Logger
from typing import Union
from transformers.trainer_utils import set_seed
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.albert.modeling_albert import AlbertForMaskedLM
from stereoset_utils import StereoSetRunner, get_experiment_id, get_logger
from stereoset_config import DataArguments, ModelArguments, TrainingArguments


def evaluate_stereoset(
    data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger
):
    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    logger.info("Set device.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Get experiment ID.")
    experiment_id = get_experiment_id(
        name="stereoset",
        model_name_or_path=model_args.model_name_or_path.split("/")[-1],
        bias_type=data_args.bias_type,
        seed=train_args.seed,
    )

    logger.info(f"Prepare model and tokenizer.")
    model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM] = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path
    )
    model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    logger.info("Start to run StereoSet.")
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Batch size: {train_args.per_device_batch_size}")
    runner = StereoSetRunner(
        model=model,
        tokenizer=tokenizer,
        model_name_or_path=model_args.model_name_or_path,
        test_data_path=data_args.test_data_path,
        per_device_batch_size=train_args.per_device_batch_size,
        max_seq_len=data_args.max_seq_length,
        bias_type=data_args.bias_type,
        device=device,
        logger=logger,
    )
    results = runner.__call__()

    logger.info(f"Save the results to: {train_args.output_dir}")
    os.makedirs(f"{train_args.output_dir}/results/stereoset", exist_ok=True)
    with open(file=f"{train_args.output_dir}/results/stereoset/{experiment_id}.json", mode="w") as fp:
        json.dump(obj=results, fp=fp, indent=4)


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    evaluate_stereoset(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
