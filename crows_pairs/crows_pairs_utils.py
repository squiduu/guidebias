import csv
import difflib
import os
import logging
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Union
from logging import Logger
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.albert.modeling_albert import AlbertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from crows_pairs_config import TrainingArguments


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


def get_experiment_id(name: str, model_name: str, bias_type: str, seed: int):
    experiment_id = f"{name}"

    # build the experiment ID
    if isinstance(model_name, str):
        experiment_id += f"_model-{model_name}"
    if isinstance(bias_type, str):
        experiment_id += f"_type-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_seed-{seed}"

    return experiment_id


class CrowsPairsRunner:
    def __init__(
        self,
        model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
        tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
        test_data_path: str,
        bias_type: str,
        device: torch.device,
        logger: Logger,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data_path = test_data_path
        self.bias_type = bias_type
        self.device = device
        self.logger = logger

    def __call__(self):
        df, bias_score = self._get_likelihood_scores()

        return df, bias_score

    def _get_likelihood_scores(self):
        """Evaluate against the CrowS-Pairs dataset using likelihood scoring."""
        df_data = self._read_test_data(self.test_data_path)

        # use GPUs
        self.model.to(self.device)

        # search each sentence
        # each row in the dataframe has sent ID and score for pro- and anti-stereo
        df = pd.DataFrame(
            columns=[
                "sent_more",
                "sent_less",
                "sent_more_score",
                "sent_less_score",
                "score",
                "stereo_antistereo",
                "bias_type",
            ]
        )

        total_stereo = 0
        total_antistereo = 0
        stereo_score = 0
        antistereo_score = 0
        num_examples = 0
        neutrals = 0
        with tqdm(total=len(df_data.index)) as p_bar:
            for _, data in df_data.iterrows():
                direction = data["direction"]
                bias = data["bias_type"]
                sent_1 = data["sent_1"]
                sent_2 = data["sent_2"]

                sent_1_token_ids = torch.Tensor.to(
                    self.tokenizer.encode(text=sent_1, return_tensors="pt"), device=self.device
                )
                sent_2_token_ids = torch.Tensor.to(
                    self.tokenizer.encode(text=sent_2, return_tensors="pt"), device=self.device
                )

                # get spans of non-changing tokens
                template_1, template_2 = _get_span(
                    seq_1=sent_1_token_ids[0], seq_2=sent_2_token_ids[0], operation="diff"
                )

                #
                score_1 = round(self._get_average_log_probability(token_ids=sent_1_token_ids, spans=template_1), 3)
                score_2 = round(self._get_average_log_probability(token_ids=sent_2_token_ids, spans=template_2), 3)

                #
                num_examples += 1
                pair_score = 0
                p_bar.update(1)
                if score_1 == score_2:
                    neutrals += 1
                else:
                    if direction == "stereo":
                        total_stereo += 1
                        if score_1 > score_2:
                            stereo_score += 1
                            pair_score = 1
                    elif direction == "antistereo":
                        total_antistereo += 1
                        if score_1 < score_2:
                            antistereo_score += 1
                            pair_score = 1

                sent_more = ""
                sent_less = ""
                if direction == "stereo":
                    sent_more = data["sent_1"]
                    sent_less = data["sent_2"]
                    sent_more_score = score_1
                    sent_less_score = score_2
                else:
                    sent_more = data["sent_2"]
                    sent_less = data["sent_1"]
                    sent_more_score = score_2
                    sent_less_score = score_1

                df = df.append(
                    {
                        "sent_more": sent_more,
                        "sent_less": sent_less,
                        "sent_more_score": sent_more_score,
                        "sent_less_score": sent_less_score,
                        "score": pair_score,
                        "stereo_antistereo": direction,
                        "bias_type": bias,
                    },
                    ignore_index=True,
                )

        self.logger.info(f"Total examples: {num_examples}")
        self.logger.info(f"Metric score: {round((stereo_score + antistereo_score) / num_examples * 100, 2)}")
        self.logger.info(f"Stereotype score: {round(stereo_score / total_stereo * 100, 2)}")
        if antistereo_score != 0:
            self.logger.info(f"Anti-stereo score: {round(antistereo_score / total_antistereo * 100, 2)}")
        self.logger.info(f"Proportion of neutrals: {round(neutrals / num_examples * 100, 2)}")

        return df, round((stereo_score + antistereo_score) / num_examples * 100, 2)

    def _read_test_data(self, test_data: str):
        """Load test data into pandas DataFrame format."""
        df_data = pd.DataFrame(columns=["sent_1", "sent_2", "direction", "bias_type"])

        with open(file=test_data, mode="r") as fp:
            csv_reader = csv.DictReader(fp)

            for row in csv_reader:
                direction = "_"
                direction = row["stereo_antistereo"]
                bias_type = row["bias_type"]

                if self.bias_type is not None and bias_type != self.bias_type:
                    continue

                sent_1 = ""
                sent_2 = ""
                if direction == "stereo":
                    sent_1 = row["sent_more"]
                    sent_2 = row["sent_less"]
                else:
                    sent_1 = row["sent_less"]
                    sent_2 = row["sent_more"]

                df_item = {"sent_1": sent_1, "sent_2": sent_2, "direction": direction, "bias_type": bias_type}
                df_data = df_data.append(df_item, ignore_index=True)

        return df_data

    def _get_average_log_probability(self, token_ids: torch.LongTensor, spans: List[int]):
        probs = []
        for position in spans:
            # mask the positions
            masked_token_ids = token_ids.clone().to(self.device)
            masked_token_ids[:, position] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs: MaskedLMOutput = self.model.forward(masked_token_ids)
                hidden_states = outputs.logits.squeeze(dim=0)[position]

            target_id = token_ids[0][position]
            log_probs = F.log_softmax(input=hidden_states, dim=0)[target_id]
            probs.append(log_probs.item())

        score = np.mean(probs)

        return score


def _get_span(seq_1: torch.LongTensor, seq_2: torch.LongTensor, operation: str):
    """Extract spans that are shared between two sequences."""
    seq_1 = [str(x) for x in seq_1.tolist()]
    seq_2 = [str(x) for x in seq_2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq_1, seq_2)
    template_1 = []
    template_2 = []
    for op in matcher.get_opcodes():
        if (operation == "equal" and op[0] == "equal") or (operation == "diff" and op[0] != "equal"):
            template_1 += [x for x in range(op[1], op[2], 1)]
            template_2 += [x for x in range(op[3], op[4], 1)]

    return template_1, template_2
