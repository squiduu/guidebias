import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from logging import Logger
from typing import Union
from torch.utils.data.dataloader import DataLoader
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.albert.modeling_albert import AlbertForMaskedLM
from stereoset_config import TrainingArguments
from stereoset_dataloader import IntrasentenceLoader

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_logger(train_args: TrainingArguments) -> Logger:
    """Create and set environments for logging.

    Args:
        args (Namespace): A parsed arguments.

    Returns:
        logger (Logger): A logger for checking progress.
    """
    # init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmtr = logging.Formatter(fmt="%(asctime)s | %(module)s | %(levelname)s > %(message)s", datefmt="%Y-%m-%d %H:%M")
    # handler for console
    console_hdlr = logging.StreamHandler()
    console_hdlr.setFormatter(fmtr)
    logger.addHandler(console_hdlr)
    # handler for .log file
    os.makedirs(train_args.output_dir, exist_ok=True)
    file_hdlr = logging.FileHandler(filename=train_args.output_dir + f"{train_args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run number: {train_args.run_name}")

    return logger


def get_experiment_id(name: str, model_name_or_path: str, bias_type: str, seed: int):
    experiment_id = f"{name}"

    # build the experiment ID
    if isinstance(model_name_or_path, str):
        experiment_id += f"_model-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_type-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_seed-{seed}"

    return experiment_id


class StereoSetRunner:
    def __init__(
        self,
        model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
        tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
        model_name_or_path: str,
        test_data_path: str,
        per_device_batch_size: int,
        max_seq_len: int,
        bias_type: str,
        device: torch.device,
        logger: Logger,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.test_data_path = test_data_path
        self.per_device_batch_size = per_device_batch_size
        self.max_seq_len = max_seq_len
        self.bias_type = bias_type
        self.device = device
        self.logger = logger

    def __call__(self):
        bias = {}

        self.logger.info("Evaluate intrasentence task.")
        bias["intrasentence"] = self._evaluate_intrasentence()

        return bias

    def _evaluate_intrasentence(self):
        sentence_probabilities = self._get_likelihood_score()

        return sentence_probabilities

    def _get_likelihood_score(self):
        self.model = self.model.to(self.device)
        pad_to_max_len = True if self.per_device_batch_size > 1 else False

        dataset = IntrasentenceLoader(
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            pad_to_max_len=pad_to_max_len,
            test_data_path=self.test_data_path,
            model_name_or_path=self.model_name_or_path,
        )

        dataloader = DataLoader(dataset=dataset, batch_size=self.per_device_batch_size)
        word_probs = defaultdict(list)

        # calculate logits for each prediction
        for sentence_id, next_token, input_ids, attention_mask, token_type_ids, target_tokens in tqdm(dataloader):
            input_ids = torch.stack(input_ids).to(self.device).transpose(0, 1)
            attention_mask = torch.stack(attention_mask).to(self.device).transpose(0, 1)
            next_token = next_token.to(self.device)
            token_type_ids = torch.stack(token_type_ids).to(self.device).transpose(0, 1)

            mask_idx = input_ids == self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model.forward(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
                )
                output = F.softmax(outputs[0], dim=-1)[mask_idx]

            output = output.index_select(dim=1, index=next_token).diag()
            for idx, item in enumerate(output):
                word_probs[sentence_id[idx]].append(item.item())

        # reconcile the probabilities into sentences
        sentence_probs = []
        for k, v in word_probs.items():
            pred = {}
            pred["id"] = k
            # score = np.sum([np.log2(i) for i in v]) + np.log2(len(v))
            score = np.mean(v)
            pred["score"] = score
            sentence_probs.append(pred)

        return sentence_probs
