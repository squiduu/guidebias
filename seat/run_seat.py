import json
import os
import pickle
import random
from argparse import Namespace
from csv import DictWriter
from logging import Logger
from seat_utils import (
    get_seat_logger,
    get_keys_to_sort_tests,
    check_availability,
    save_encoded_vectors,
    set_seed,
    prepare_model_and_tokenizer,
    get_encodings,
)
from seat_config import get_seat_args, TEST_EXT
import weat


def run_seat(args: Namespace, logger: Logger):
    """Parse args for seat to run and which models to evaluate.

    Args:
        args (Namespace): A parsed arguments.
        logger (Logger): A logger for checking process.
    """
    # set seed
    seed = random.randint(0, 100)
    logger.info(f"Seed: {seed}")
    set_seed(seed)

    # get all tests
    all_tests = sorted(
        [
            entry[: -len(TEST_EXT)]
            for entry in os.listdir(args.data_dir)
            if not entry.startswith(".") and entry.endswith(TEST_EXT)
        ],
        key=get_keys_to_sort_tests,
    )
    logger.info(f"Found tests: {all_tests}")

    # check the available tests
    tests = (
        check_availability(arg_str=args.tests, allowed_set=all_tests, item_type="test")
        if args.tests is not None
        else all_tests
    )
    logger.info(f"Selected tests: {tests}")

    # check the available models
    available_models = (
        check_availability(arg_str=args.model_name, allowed_set=["bert", "roberta", "albert"], item_type="model")
        if args.model_name is not None
        else ["bert", "roberta", "albert"]
    )

    logger.info(f"Selected models: {available_models}")
    results = []
    for model_name in available_models:
        logger.info(f"Start to run the SEAT for {model_name}.")
        # load the model and tokenizer
        model, tokenizer = prepare_model_and_tokenizer(version=args.version, args=args)

        for test in tests:
            # set enc path
            enc_path = f"{args.enc_save_dir}{args.model_name}_{test}"
            enc_path += "_debiased.pkl" if args.debiased else "_biased.pkl"

            # get test data
            test_data = json.load(fp=open(os.path.join(args.data_dir, f"{test}{TEST_EXT}"), mode="r"))

            # get encodings
            encs_targ1, encs_targ2, encs_attr1, encs_attr2 = get_encodings(
                data_keys=["targ1", "targ2", "attr1", "attr2"], data=test_data, model=model, tokenizer=tokenizer
            )

            # save encoded vectors in `test_data` with `data` key name
            encoded_data = save_encoded_vectors(
                data=test_data,
                encs_targ1=encs_targ1,
                encs_targ2=encs_targ2,
                encs_attr1=encs_attr1,
                encs_attr2=encs_attr2,
            )
            if args.cache_encs:
                with open(file=enc_path, mode="wb") as cache_encs_fp:
                    pickle.dump(obj=encoded_data, file=cache_encs_fp)

            # get WEAT results and save them as a result dict
            effect_size, delta_mean, stdev, p_value = weat.run_test(
                encs=encoded_data, num_samples=args.num_samples, use_parametric=args.use_parametric, logger=logger
            )
            logger.info(f"Test: {test} - Effect Size: {round(effect_size, 3)}")
            results.append(
                dict(
                    version=args.version,
                    test=test,
                    p_value=round(p_value, 9),
                    effect_size=round(effect_size, 3),
                    delta_mean=round(delta_mean, 9),
                    stdev=round(stdev, 9),
                    avg_abs_effect_size=None,
                )
            )

        #
        avg_abs_effect_size = {
            "avg_abs_effect_size": sum([abs(results[i]["effect_size"]) for i in range(len(results))]) / len(results)
        }

        #
        for result in results:
            logger.info(
                "Test: {test}\tp-value: {p_value:.9f}\tdelta_mean: {delta_mean:.9f}\tstdev: {stdev:.9f}\teffect-size: {effect_size:.3f}".format(
                    **result
                )
            )
        logger.info(f"Avg. abs. effect-size: {round(avg_abs_effect_size['avg_abs_effect_size'], 3)}")

    #
    logger.info(f"Save the SEAT results to {args.output_dir}{args.model_name}_seat_{args.run_name}.csv")

    #
    results[-1]["avg_abs_effect_size"] = round(avg_abs_effect_size["avg_abs_effect_size"], 3)

    #
    with open(file=f"{args.output_dir + args.model_name}_seat_{args.run_name}.csv", mode="w") as res_fp:
        writer = DictWriter(f=res_fp, fieldnames=dict.keys(results[0]), delimiter="\t")
        writer.writeheader()
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    args = get_seat_args()
    logger = get_seat_logger(args)

    run_seat(args=args, logger=logger)
