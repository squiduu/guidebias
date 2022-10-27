import json
import os
import torch
from typing import Union
from logging import Logger
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer_utils import set_seed
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.albert.modeling_albert import AlbertForMaskedLM
from crows_pairs_config import DataArguments, ModelArguments, TrainingArguments
from crows_pairs_utils import get_logger, get_experiment_id, CrowsPairsRunner


def evaluate_crows_pairs(
    data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger
):
    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    logger.info("Set device.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Get experiment ID.")
    experiment_id = get_experiment_id(
        name="crows",
        model_name=model_args.model_name,
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

    logger.info("Start to run CrowS-Pairs benchmark.")
    logger.info(f"Checkpoint: {model_args.model_name_or_path}")
    logger.info(f"Bias type: {data_args.bias_type}")
    runner = CrowsPairsRunner(
        model=model,
        tokenizer=tokenizer,
        test_data_path=data_args.test_data_path,
        bias_type=data_args.bias_type,
        device=device,
        logger=logger,
    )
    df, bias_score = runner.__call__()

    logger.info(f"Save the results to: {train_args.output_dir}")
    os.makedirs(f"{train_args.output_dir}/results/crows", exist_ok=True)
    with open(file=f"{train_args.output_dir}/results/crows/{experiment_id}.json", mode="w") as fp:
        json.dump(obj=bias_score, fp=fp, indent=4)

    df.to_csv(f"{train_args.output_dir}/results/crows/{experiment_id}.csv")


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    evaluate_crows_pairs(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
