import torch
import torch.cuda.amp as amp
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from logging import Logger
from torch.optim.adamw import AdamW
from transformers.trainer_utils import set_seed
from transformers.hf_argparser import HfArgumentParser
from config import DataArguments, ModelArguments, TrainingArguments
from utils import (
    clear_console,
    get_bias_loss,
    get_hidden_states,
    get_lm_loss,
    get_logger,
    get_batch_data,
    get_words,
    make_inputs,
    prepare_models_and_tokenizer,
    prepare_stereo_sents,
    prepare_neutral_sents,
    JSDivergence,
)


def finetune(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
    """Fine-tune a pre-trained BERT for debiasing.

    Args:
        data_args (DataArguments): A parsed data arguments.
        model_args (ModelArguments): A parsed model arguments.
        train_args (TrainingArguments): A parsed training arguments.
        logger (Logger): A logger for checking progress information.
    """
    logger.info(f"Set device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Set seed: {train_args.seed}")
    set_seed(train_args.seed)

    logger.info(f"Prepare models and tokenizer: {model_args.model_name}")
    guide, trainee, tokenizer = prepare_models_and_tokenizer(model_args=model_args)
    guide.to(device)
    trainee.to(device)

    logger.info(f"Set model and optimizer with APEX.")
    optimizer = AdamW(params=trainee.parameters(), lr=train_args.lr)
    scaler = amp.grad_scaler.GradScaler()

    logger.info("Prepare words for sentence generation.")
    MALE_WORDS, FEMALE_WORDS, STEREO_WORDS, WIKI_WORDS = get_words(data_args=data_args)

    logger.info("Prepare stereotype sentences.")
    male_sents = prepare_stereo_sents(
        gender_words=MALE_WORDS, wiki_words=WIKI_WORDS[: data_args.num_stereo_wiki_words], stereo_words=STEREO_WORDS
    )
    female_sents = prepare_stereo_sents(
        gender_words=FEMALE_WORDS, wiki_words=WIKI_WORDS[: data_args.num_stereo_wiki_words], stereo_words=STEREO_WORDS,
    )
    logger.info("Prepare non-stereotype sentences.")
    neutral_sents = prepare_neutral_sents(gender_words=MALE_WORDS + FEMALE_WORDS, wiki_words=WIKI_WORDS)
    neutral_sents = neutral_sents[: len(male_sents)]

    logger.info("Prepare a train dataloader.")
    dl = DataLoader(
        dataset=[i for i in range(len(male_sents))],
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    num_training_steps = int(train_args.num_epochs * len(dl))
    num_warmup_steps = int(num_training_steps * train_args.warmup_proportion)
    logger.info(f"Set lr scheduler with {num_warmup_steps} warm-up steps.")
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    jsd_runner = JSDivergence(reduction="batchmean")

    #
    for ep in range(1, int(train_args.num_epochs) + 1):
        optimizer.zero_grad()

        # start to fine-tune
        dl = tqdm(dl)
        for iter, batch_idx in enumerate(dl):
            # get as many input sentences as the number of batch size
            male_sents_batch, female_sents_batch, neutral_sents_batch = get_batch_data(
                batch_idx=batch_idx, male_sents=male_sents, female_sents=female_sents, neutral_sents=neutral_sents
            )

            # tokenize and send to cuda for model inputs
            male_inputs, female_inputs, neutral_inputs = make_inputs(
                male_sents=male_sents_batch,
                female_sents=female_sents_batch,
                neutral_sents=neutral_sents_batch,
                tokenizer=tokenizer,
                device=device,
            )

            with amp.autocast_mode.autocast():
                # get sequence directional averaged hidden states in the last layer of BERT encoder
                (
                    guide_neutral_hidden,
                    male_stereo_hidden,
                    female_stereo_hidden,
                    trainee_neutral_hidden,
                ) = get_hidden_states(
                    guide=guide,
                    trainee=trainee,
                    male_inputs=male_inputs,
                    female_inputs=female_inputs,
                    neutral_inputs=neutral_inputs,
                    layer_number=-1,
                    dim=1,
                )

                # get bias loss for stereotype sentences
                bias_loss = get_bias_loss(
                    jsd_runner=jsd_runner, hidden_1=male_stereo_hidden, hidden_2=female_stereo_hidden
                )
                # get language modeling loss for non-stereotype sentences
                lm_loss = get_lm_loss(hidden_1=trainee_neutral_hidden, hidden_2=guide_neutral_hidden)
                loss = train_args.debias_ratio * bias_loss + (1 - train_args.debias_ratio) * lm_loss

            scaler.scale(loss / train_args.grad_accum_steps).backward()
            scale = scaler.get_scale()
            skip_scheduler = scale != scaler.get_scale()
            if not skip_scheduler:
                scheduler.step()
            if (iter + 1) % train_args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # check fine-tuning procedure
            dl.set_description(
                f"Epoch: {ep}/{int(train_args.num_epochs)} - Loss: {loss:.4f} - Bias Loss: {bias_loss:.4f} - LM Loss: {lm_loss:.4f}"
            )

    logger.info("Save a fine-tuned model.")
    trainee.save_pretrained(
        f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}_seed{train_args.seed}_num{data_args.num_wiki_words}-{data_args.num_stereo_wiki_words}"
    )
    tokenizer.save_pretrained(
        f"./out/{model_args.model_name}_{train_args.run_name}_ep{ep}_seed{train_args.seed}_num{data_args.num_wiki_words}-{data_args.num_stereo_wiki_words}"
    )


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    finetune(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
