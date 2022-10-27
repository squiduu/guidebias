import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from logging import Logger
from typing import List, Tuple
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel
from transformers.tokenization_utils_base import BatchEncoding
from config import DataArguments, ModelArguments, TrainingArguments


def clear_console():
    # default to Ubuntu
    command = "clear"
    # if machine is running on Windows
    if os.name in ["nt", "dos"]:
        command = "cls"
    os.system(command)


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
    file_hdlr = logging.FileHandler(filename=train_args.output_dir + f"swit_{train_args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run name: {train_args.run_name}")

    return logger


def prepare_models_and_tokenizer(model_args: ModelArguments) -> Tuple[BertForMaskedLM, BertForMaskedLM, BertTokenizer]:
    # get tokenizer regardless of model version
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    freezed_model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
    freezed_encoder = BertModel.from_pretrained(model_args.model_name_or_path)
    tuning_model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
    tuning_encoder = BertModel.from_pretrained(model_args.model_name_or_path)

    #
    freezed_model.bert = freezed_encoder
    tuning_model.bert = tuning_encoder

    #
    freezed_model.cuda()
    tuning_model.cuda()

    return freezed_model, tuning_model, tokenizer


def get_words(data_args: DataArguments) -> Tuple[List[str], List[str], List[str]]:
    with open(file=f"./data/male/male_words_{data_args.num_gender_words}.json", mode="r") as male_fp:
        MALE_WORDS = json.load(male_fp)
    MALE_WORDS = MALE_WORDS[: data_args.num_gender_words]
    with open(file=f"./data/female/female_words_{data_args.num_gender_words}.json", mode="r") as female_fp:
        FEMALE_WORDS = json.load(female_fp)
    FEMALE_WORDS = FEMALE_WORDS[: data_args.num_gender_words]

    with open(file=f"./data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        STEREO_WORDS = json.load(ster_fp)

    with open(file=f"./data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI_WORDS = json.load(wiki_fp)
    WIKI_WORDS = filter_wiki(wiki_words=WIKI_WORDS, gender_words=MALE_WORDS + FEMALE_WORDS, stereo_words=STEREO_WORDS)
    WIKI_WORDS = WIKI_WORDS[: data_args.num_wiki_words]

    return MALE_WORDS, FEMALE_WORDS, STEREO_WORDS, WIKI_WORDS


def filter_wiki(wiki_words: List[str], gender_words: List[str], stereo_words: List[str]):
    filtered = []
    for word in wiki_words:
        if word not in (gender_words + stereo_words):
            filtered.append(word)

    return filtered


def prepare_stereo_sents(gender_words: List[str], wiki_words: List[str], stereo_words: List[str]) -> List[str]:
    sents = []
    for i in range(len(gender_words)):
        for j in range(len(wiki_words)):
            for k in range(len(stereo_words)):
                sents.append(gender_words[i] + " " + wiki_words[j] + " " + stereo_words[k] + " .")

    return sents


def prepare_neutral_sents(gender_words: List[str], wiki_words: List[str]) -> List[str]:
    sents = []
    for i in range(len(gender_words)):
        for j in range(len(wiki_words)):
            for k in range(len(wiki_words)):
                sents.append(gender_words[i] + " " + wiki_words[j] + " " + wiki_words[k] + " .")

    return sents


class JSDivergence(nn.Module):
    def __init__(self, reduction: str = "batchmean") -> None:
        """Get average JS-Divergence between two networks.

        Args:
            dim (int, optional): A dimension along which softmax will be computed. Defaults to 1.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "batchmean".
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, hidden_1: torch.FloatTensor, hidden_2: torch.FloatTensor) -> torch.FloatTensor:
        h1 = F.softmax(hidden_1, dim=1)
        h2 = F.softmax(hidden_2, dim=1)

        avg_hidden = (h1 + h2) / 2.0

        jsd = 0.0
        jsd += F.kl_div(input=F.log_softmax(hidden_1, dim=1), target=avg_hidden, reduction=self.reduction)
        jsd += F.kl_div(input=F.log_softmax(hidden_2, dim=1), target=avg_hidden, reduction=self.reduction)

        return jsd / 2.0


def get_batch_data(
    batch_idx: torch.LongTensor, male_sents: List[str], female_sents: List[str], neutral_sents: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    male_sents_batch = []
    female_sents_batch = []
    neutral_sents_batch = []

    for i in batch_idx:
        male_sents_batch.append(male_sents[torch.Tensor.item(i)])
        female_sents_batch.append(female_sents[torch.Tensor.item(i)])
        neutral_sents_batch.append(neutral_sents[torch.Tensor.item(i)])

    return male_sents_batch, female_sents_batch, neutral_sents_batch


def make_inputs(
    male_sents: List[str],
    female_sents: List[str],
    neutral_sents: List[str],
    tokenizer: BertTokenizer,
    device: torch.device,
):
    male_inputs = tokenizer(text=male_sents, padding=True, truncation=True, return_tensors="pt")
    female_inputs = tokenizer(text=female_sents, padding=True, truncation=True, return_tensors="pt")
    neutral_inputs = tokenizer(text=neutral_sents, padding=True, truncation=True, return_tensors="pt")
    #
    for key in male_inputs.keys():
        male_inputs[key] = torch.Tensor.cuda(male_inputs[key], device=device)
        female_inputs[key] = torch.Tensor.cuda(female_inputs[key], device=device)
        neutral_inputs[key] = torch.Tensor.cuda(neutral_inputs[key], device=device)

    return male_inputs, female_inputs, neutral_inputs


def get_hidden_states(
    guide: BertForMaskedLM,
    trainee: BertForMaskedLM,
    male_inputs: BatchEncoding,
    female_inputs: BatchEncoding,
    neutral_inputs: BatchEncoding,
    layer_number: int,
    dim: int,
):
    # with torch.no_grad():
    guide_neutral_outputs = guide.forward(**neutral_inputs, output_hidden_states=True)
    trainee_male_outputs = trainee.forward(**male_inputs, output_hidden_states=True)
    trainee_female_outputs = trainee.forward(**female_inputs, output_hidden_states=True)
    trainee_neutral_outputs = trainee.forward(**neutral_inputs, output_hidden_states=True)

    #
    guide_neutral_hidden = guide_neutral_outputs.hidden_states[layer_number].mean(dim=dim)
    male_stereo_hidden = trainee_male_outputs.hidden_states[layer_number].mean(dim=dim)
    female_stereo_hidden = trainee_female_outputs.hidden_states[layer_number].mean(dim=dim)
    trainee_neutral_hidden = trainee_neutral_outputs.hidden_states[layer_number].mean(dim=dim)

    return guide_neutral_hidden, male_stereo_hidden, female_stereo_hidden, trainee_neutral_hidden


def get_bias_loss(jsd_runner: JSDivergence, hidden_1: torch.FloatTensor, hidden_2: torch.FloatTensor):
    bias_hidden_jsd = jsd_runner.forward(hidden_1=hidden_1, hidden_2=hidden_2)
    bias_hidden_cossim = F.cosine_similarity(hidden_1, hidden_2).mean()

    return bias_hidden_jsd - bias_hidden_cossim


def get_lm_loss(hidden_1: torch.FloatTensor, hidden_2: torch.FloatTensor):
    lm_hidden_kld = F.kl_div(
        input=F.log_softmax(hidden_1, dim=-1), target=F.softmax(hidden_2, dim=-1), reduction="batchmean"
    )
    lm_hidden_cossim = F.cosine_similarity(hidden_1, hidden_2).mean()

    return lm_hidden_kld - lm_hidden_cossim
