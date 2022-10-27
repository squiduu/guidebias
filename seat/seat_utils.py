import logging
import os
import json
import random
import numpy as np
import torch
import re
from logging import Logger
from argparse import Namespace
from typing import Dict, List, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.albert.modeling_albert import AlbertModel

CATEGORY = "category"


def get_seat_logger(args: Namespace) -> Logger:
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
    os.makedirs(args.output_dir, exist_ok=True)
    file_hdlr = logging.FileHandler(filename=args.output_dir + f"seat_{args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run number: {args.run_name}")
    # record arguments and hparams
    logger.info(f"Config: {vars(args)}")

    return logger


def get_keys_to_sort_tests(test: str):
    """Return tuple to be used as a sort key for the specified test name.
    Break test name into pieces consisting of the integers in the name and the strings in between them."""
    key = ()
    prev_end = 0
    for match in re.finditer(r"\d+", test):
        key = key + (test[prev_end : match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key


def check_availability(arg_str: str, allowed_set: list, item_type: str):
    """Given a comma-separated string of items, split on commas and check if all items are in `allowed_set`."""
    test_items = arg_str.split(",")
    for test_item in test_items:
        if test_item not in allowed_set:
            raise ValueError(f"Unknown {item_type}: {test_item}.")

    return test_items


def save_encoded_vectors(data: dict, encs_targ1: dict, encs_targ2: dict, encs_attr1: dict, encs_attr2: dict):
    """Save the encoded vectors in the dataset with another key name."""
    data["targ1"]["encs"] = encs_targ1
    data["targ2"]["encs"] = encs_targ2
    data["attr1"]["encs"] = encs_attr1
    data["attr2"]["encs"] = encs_attr2

    return data


def load_json(json_path: str):
    """Load from .json file. We expect a certain format later, so do some post processing."""
    all_data = json.load(open(file=json_path, mode="r"))
    data = {}
    for key, value in dict.items(all_data):
        examples = value["examples"]
        data[key] = examples
        value["examples"] = examples

    return all_data


def set_seed(seed: int):
    """Set a seed for complete reproducibility.

    Args:
        seed (int): A random integer for seed.
    """
    # for python
    random.seed(seed)
    # for numpy
    np.random.seed(seed)
    # for torch
    torch.manual_seed(seed)
    # for cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_model_and_tokenizer(
    version: str, args: Namespace
) -> Tuple[Union[BertModel, RobertaModel, AlbertModel], Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],]:
    """Load a pre-trained or checkpoint model as evaluation and corresponding tokenizer.

    Args:
        version (str): A version of pre-trained model.
        args (Namespace): A parsed arguments.

    Returns:
        model (Union[BertModel, RobertaModel, AlbertModel]): A pre-trained or checkpoint model.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    if "bert" in version:
        model_class = BertModel
        tokenizer_class = BertTokenizer
    elif "roberta" in version:
        model_class = RobertaModel
        tokenizer_class = RobertaTokenizer
    else:
        model_class = AlbertModel
        tokenizer_class = AlbertTokenizer

    # get tokenizer regardless of version
    tokenizer = tokenizer_class.from_pretrained(args.version)

    # get model
    if args.use_ckpt:
        model = model_class.from_pretrained(args.ckpt_dir)
    else:
        model = model_class.from_pretrained(args.version)

    model.eval()

    return model, tokenizer


def get_encodings(
    data_keys: List[str],
    data: Dict[str, str],
    model: Union[BertModel, RobertaModel, AlbertModel],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Encode the input data using the PreTrainedTokenizer and PreTrainedModel.

    Args:
        data_keys (List[str]): Key names for iteration.
        data (Dict[str, str]): An input data.
        model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A Pre-trained PreTrainedModel model.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A Pre-trained PreTrainedTokenizer.

    Returns:
        encs_targ1, encs_targ2, encs_attr1, encs_attr2: Encodings corresponding to the encoderd and tokenizer.
    """
    for data_key in data_keys:
        encs = {}
        for sent in data[data_key]["examples"]:
            inputs = tokenizer(sent, return_tensors="pt")

            # get encoder outputs
            outputs: BaseModelOutputWithPoolingAndCrossAttentions = model.forward(**inputs)

            # extract the last hidden state
            encs[sent] = outputs.last_hidden_state.mean(dim=1).detach().reshape(-1).numpy()
            encs[sent] /= np.linalg.norm(encs[sent])

        if data_key == "targ1":
            encs_targ1 = encs
        elif data_key == "targ2":
            encs_targ2 = encs
        elif data_key == "attr1":
            encs_attr1 = encs
        else:
            encs_attr2 = encs

    return encs_targ1, encs_targ2, encs_attr1, encs_attr2
