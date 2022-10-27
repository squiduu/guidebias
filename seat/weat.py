""" Implements the WEAT tests """
from logging import Logger
import math
import itertools as it
from typing import Dict
import numpy as np
import scipy.special
import scipy.stats
from seat_config import get_seat_args
from seat_utils import get_seat_logger

# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.


def get_cosine_similarity(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def construct_cosine_similarity_lookup(XY, AB):
    """
    Args:
        XY: mapping from target string to target vector (either in X or Y)
        AB: mapping from attribute string to attribute vectore (either in A or B)

    Returns:
        An array of size (len(XY), len(AB)) containing cosine similarities between items in XY and items in AB.
    """

    cos_sims = np.zeros((len(XY), len(AB)))
    for xy in XY:
        for ab in AB:
            cos_sims[xy, ab] = get_cosine_similarity(XY[xy], AB[ab])

    return cos_sims


def s_wAB(A, B, cos_sims):
    """
    Returns:
        Vector of s(w, A, B) across w, where s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    """
    return cos_sims[:, A].mean(axis=1) - cos_sims[:, B].mean(axis=1)


def s_XAB(X, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    return slightly more computationally efficient version of WEAT
    statistic for p-value computation.

    Caliskan defines the WEAT statistic s(X, Y, A, B) as
        sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
    where s(w, A, B) is defined as
        mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    The p-value is computed using a permutation test on (X, Y) over all
    partitions (X', Y') of X union Y with |X'| = |Y'|.

    However, for all partitions (X', Y') of X union Y,
        s(X', Y', A, B)
      = sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = C,
    a constant.  Thus
        sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = sum_{x in X'} s(x, A, B) + (C - sum_{x in X'} s(x, A, B))
      = C + 2 sum_{x in X'} s(x, A, B).

    By monotonicity,
        s(X', Y', A, B) > s(X, Y, A, B)
    if and only if
        [s(X', Y', A, B) - C] / 2 > [s(X, Y, A, B) - C] / 2,
    that is,
        sum_{x in X'} s(x, A, B) > sum_{x in X} s(x, A, B).
    Thus we only need use the first component of s(X, Y, A, B) as our
    test statistic.
    """
    return s_wAB_memo[X].sum()


def s_XYAB(X, Y, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    the WEAT test statistic for p-value computation.
    """
    return s_XAB(X, s_wAB_memo) - s_XAB(Y, s_wAB_memo)


def p_value_permutation_test(X, Y, A, B, n_samples, cossims, parametric: bool, logger: Logger):
    """Compute the p-val for the permutation test, which is defined as
    the probability that a random even partition X_i, Y_i of X u Y
    satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    """
    X = np.array(list(X), dtype=np.int)
    Y = np.array(list(Y), dtype=np.int)
    A = np.array(list(A), dtype=np.int)
    B = np.array(list(B), dtype=np.int)

    assert len(X) == len(Y)
    size = len(X)
    s_wAB_memo = s_wAB(A, B, cos_sims=cossims)
    XY = np.concatenate((X, Y))

    if parametric:
        # logger.info("Using parametric test.")
        s = s_XYAB(X, Y, s_wAB_memo)

        samples = []
        for _ in range(n_samples):
            np.random.shuffle(XY)
            Xi = XY[:size]
            Yi = XY[size:]
            assert len(Xi) == len(Yi)
            si = s_XYAB(Xi, Yi, s_wAB_memo)
            samples.append(si)

        # Compute sample standard deviation and compute p-value by assuming normality of null distribution
        logger.info("Infer p-value based on normal distribution.")
        (shapiro_test_stat, shapiro_p_val) = scipy.stats.shapiro(samples)
        logger.info(
            "Shapiro-Wilk normality test statistic: {:.2g}, p-value: {:.2g}".format(shapiro_test_stat, shapiro_p_val)
        )
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)
        logger.info("Sample mean: {:.2g}, sample standard deviation: {:.2g}".format(sample_mean, sample_std))
        p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
        return p_val

    else:
        # logger.info("Use non-parametric test.")
        s = s_XAB(X, s_wAB_memo)
        total_true = 0
        total_equal = 0
        total = 0

        num_partitions = int(scipy.special.binom(2 * len(X), len(X)))
        if num_partitions > n_samples:
            # We only have as much precision as the number of samples drawn;
            # bias the p-value (hallucinate a positive observation) to
            # reflect that.
            total_true += 1
            total += 1
            # logger.info("Draw {} samples (and biasing by 1).".format(n_samples - total))
            for _ in range(n_samples - 1):
                np.random.shuffle(XY)
                Xi = XY[:size]
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        else:
            logger.info("Use exact test ({} partitions).".format(num_partitions))
            for Xi in it.combinations(XY, len(X)):
                Xi = np.array(Xi, dtype=np.int)
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        if total_equal:
            logger.warning("Equalities contributed {}/{} to p-value.".format(total_equal, total))

        return total_true / total


def mean_s_wAB(X, A, B, cos_sims):
    return np.mean(s_wAB(A, B, cos_sims[X]))


def stdev_s_wAB(X, A, B, cos_sims):
    return np.std(s_wAB(A, B, cos_sims[X]), ddof=1)


def get_effect_size(X, Y, A, B, cos_sims):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X = list(X)
    Y = list(Y)
    A = list(A)
    B = list(B)

    numerator = mean_s_wAB(X, A, B, cos_sims=cos_sims) - mean_s_wAB(Y, A, B, cos_sims=cos_sims)
    denominator = stdev_s_wAB(X + Y, A, B, cos_sims=cos_sims)

    return numerator / denominator, numerator, denominator


def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (_, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (_, v)) in enumerate(Y.items())),
    )


def run_test(encs: Dict[str, np.ndarray], num_samples: int, use_parametric: bool, logger: Logger):
    """Run a WEAT.

    Args:
        encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2 to dictionaries containing the category and the encodings.
        n_samples (int): number of samples to draw to estimate p-value (use exact test if number of permutations is less than or equal to n_samples).
    """
    X, Y = encs["targ1"]["encs"], encs["targ2"]["encs"]
    A, B = encs["attr1"]["encs"], encs["attr2"]["encs"]

    # first convert all keys to ints to facilitate array lookups
    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    cos_sims = construct_cosine_similarity_lookup(XY, AB)
    p_value = p_value_permutation_test(
        X, Y, A, B, num_samples, cossims=cos_sims, parametric=use_parametric, logger=logger
    )
    effect_size, delta_mean, stdev = get_effect_size(X, Y, A, B, cos_sims=cos_sims)
    # logger.info("Effect-size: %g", effect_size)

    return effect_size, delta_mean, stdev, p_value


if __name__ == "__main__":
    X = {"x" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    Y = {"y" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    A = {"a" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    B = {"b" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    A = X
    B = Y

    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    args = get_seat_args()
    logger = get_seat_logger(args)

    cossims = construct_cosine_similarity_lookup(XY, AB)
    logger.info("Compute p-values.")
    pval = p_value_permutation_test(X, Y, A, B, cossims=cossims, n_samples=10000)
    logger.info("p-value: %g", pval)

    logger.info("Compute effect-sizes.")
    esize = get_effect_size(X, Y, A, B, cos_sims=cossims)
    logger.info("effect-size: %g", esize)
