from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them.
    """

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    num_gender_words: Optional[int] = field(default=0, metadata={"help": "The number of gender words."})
    num_wiki_words: Optional[int] = field(default=0, metadata={"help": "The number of wiki words."})
    num_stereo_wiki_words: Optional[int] = field(default=0, metadata={"help": "The number of wiki words for stereo."})


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class TrainingArguments:
    """It is the subset of the arguments we use in our example scripts which relate to the training loop itself."""

    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    num_gpus: Optional[int] = field(default=1, metadata={"help": "The number of GPUs for training."})
    batch_size: Optional[int] = field(default=1, metadata={"help": "The number of batch size per device."})
    project: Optional[str] = field(default=None, metadata={"help": "A project name."})
    run_name: Optional[str] = field(default=None, metadata={"help": "An optional name of each run."})
    seed: Optional[int] = field(default=42, metadata={"help": "A seed number."})
    lr: Optional[float] = field(default=2e-5, metadata={"help": "A learning rate for training."})
    num_epochs: Optional[int] = field(default=1, metadata={"help": "A maximum number of epochs."})
    num_workers: Optional[int] = field(default=0, metadata={"help": "A number of workers for dataloader."})
    grad_accum_steps: Optional[int] = field(default=1, metadata={"help": "A number of accumulation steps."})
    warmup_proportion: Optional[float] = field(default=0.0, metadata={"help": "A warm-up proportion for scheduler."})
    debias_ratio: Optional[float] = field(default=0.0, metadata={"help": "A debiasing ratio for objective."})
