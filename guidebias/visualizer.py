import os
import json, pickle
import random
import nltk
import numpy as np
import matplotlib.pyplot as plt
from pattern3.text.en.inflect import pluralize, singularize
from transformers.trainer_utils import set_seed
from sklearn.manifold import TSNE
from numpy import linalg

DATA_DIR = "../data/tests/"


def load_json(file):
    """Load from json. We expect a certain format later, so do some post processing"""
    all_data = json.load(open(file=file, mode="r"))
    data = {}
    for k, v in dict.items(all_data):
        examples = v["examples"]
        data[k] = examples

    return all_data


def my_pluralize(word):
    if word in ["he", "she", "her", "hers"]:
        return word
    if word == "brother":
        return "brothers"
    if word == "drama":
        return "dramas"

    return pluralize(word)


def my_singularize(word):
    if word in ["hers", "his", "theirs"]:
        return word

    return singularize(word)


def match_one_test(test_name):
    # load words
    word_filename = "weat{}.jsonl".format(test_name)
    word_file = os.path.join(DATA_DIR, word_filename)
    word_data = load_json(word_file)

    # load simple sentences
    sent_filename = "seat{}.jsonl".format(test_name)
    sent_file = os.path.join(DATA_DIR, sent_filename)
    sent_data = load_json(sent_file)

    word2sents = dict()
    for key in ["targ1", "targ2", "attr1", "attr2"]:
        words = word_data[key]["examples"]
        for word in words:
            word2sents[word] = []
    all_words = set(word2sents.keys())
    print(" ----- All words. ----- ")
    print(all_words)

    unmatched_sents = []
    for key in ["targ1", "targ2", "attr1", "attr2"]:
        sents = sent_data[key]["examples"]
        for sent in sents:
            matched = False
            for word in all_words:
                word_ = str.lower(word)
                sent_ = str.lower(sent)
                tokens = nltk.word_tokenize(sent_)
                word_variants = set({word})
                word_variants.add(my_pluralize(word_))
                word_variants.add(my_singularize(word_))
                matched_words = []
                for word_variant in word_variants:
                    if word_variant in tokens:
                        matched_words.append(word)
                        if matched:
                            print("'{}' is matched to {}!.".format(sent, word))
                            print(matched_words)
                        matched = True
                        list.append(word2sents[word], sent)
                        break

            if not matched:
                unmatched_sents.append(sent)

    with open(os.path.join(DATA_DIR, "word2sents{}.jsonl".format(test_name)), "w") as outfile:
        json.dump(word2sents, outfile)

    print(" ----- Unmatched sentences: {} ----- ".format(unmatched_sents))


def match():
    for test_name in ["6", "6b", "7", "7b", "8", "8b"]:
        match_one_test(f"{test_name}")


def get_sentence_vectors(test_name: str, debiased: bool, model_name: str):
    if debiased:
        sent_encs_filename = f"{model_name}_seat{test_name}_debiased.pkl"
    else:
        sent_encs_filename = f"{model_name}_seat{test_name}_biased.pkl"

    #
    file = open(file=os.path.join(DATA_DIR, sent_encs_filename), mode="rb")
    data = pickle.load(file)

    #
    all_sent_vectors = dict()
    for key in ["targ1", "targ2", "attr1", "attr2"]:
        for text in data[key]["examples"]:
            all_sent_vectors[text] = data[key]["encs"][text]

    print(f"Loaded sentence vectors for SEAT-{test_name}.")

    return all_sent_vectors


def plot_tsne(word_vectors: dict, perplexity: str, filename: str, do_tsne: bool):
    #
    words = [
        "male",
        "man",
        "boy",
        "brother",
        "he",
        "him",
        "his",
        "son",
        "female",
        "woman",
        "girl",
        "sister",
        "she",
        "her",
        "hers",
        "daughter",
        "career",
        "family",
        "dance",
        "math",
        "art",
        "literature",
        "science",
        "technology",
    ]
    X = np.array([word_vectors[word] for word in words])

    # t-SNE
    if do_tsne:
        X_embedded = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
        X1, X2 = X_embedded[:, 0], X_embedded[:, 1]
    else:
        X1, X2 = X[:, 0], X[:, 1]

    #
    color_dict = {
        "male": "tab:blue",
        "man": "tab:blue",
        "boy": "tab:blue",
        "brother": "tab:blue",
        "he": "tab:blue",
        "him": "tab:blue",
        "his": "tab:blue",
        "son": "tab:blue",
        "female": "tab:red",
        "woman": "tab:red",
        "girl": "tab:red",
        "sister": "tab:red",
        "she": "tab:red",
        "her": "tab:red",
        "hers": "tab:red",
        "daughter": "tab:red",
    }

    #
    colors = [color_dict.get(word, "k") for word in words]
    plt.style.use("seaborn")
    plt.scatter(X1, X2, color=colors)
    for word_id, word in enumerate(words):
        plt.annotate(word, (X1[word_id], X2[word_id]), fontsize=20)
    plt.tight_layout()
    x_margin = (max(X1) - min(X1)) * 0.1
    y_margin = (max(X2) - min(X2)) * 0.1
    plt.xlim(min(X1) - x_margin, max(X1) + x_margin)
    plt.ylim(min(X2) - y_margin, max(X2) + y_margin)
    plt.xticks([])
    plt.yticks([])
    plot_dir = "./out/"
    filename = os.path.join(plot_dir, f"tsne")
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f" ---- Save to {filename}. ----- ")
    plt.savefig(filename)
    plt.clf()


def get_words_from_sentences(word2sents: dict, test_name: str, debiased: bool, model_name: str):
    all_sent_vectors = get_sentence_vectors(test_name=test_name, debiased=debiased, model_name=model_name)

    word_vectors = dict()
    for word in word2sents:
        sents = word2sents[word]
        sent_vectors = np.array([all_sent_vectors[sent] for sent in sents])
        sent_vectors = sent_vectors / linalg.norm(sent_vectors, ord=2, axis=-1, keepdims=True)
        word_vector = np.mean(sent_vectors, axis=0)
        word_vector = word_vector / linalg.norm(word_vector, ord=2, axis=-1, keepdims=True)
        word_vectors[word] = word_vector

    return word_vectors


def visualize_few_words(debiased: bool, do_tsne: bool, perplexity: int, use_sents: bool, model_name: str):
    #
    bias_flag = "debiased" if debiased else "biased"
    sent_flag = "sent" if use_sents else "word"

    #
    all_word_vectors = dict()
    for test_name in ["6", "6b", "7", "7b", "8", "8b"]:
        filename = os.path.join(DATA_DIR, f"word2sents{test_name}.jsonl")
        word2sents = json.load(open(file=filename, mode="r"))

        #
        word_vectors = get_words_from_sentences(
            word2sents=word2sents, test_name=test_name, debiased=debiased, model_name=model_name
        )
        all_word_vectors.update(word_vectors)

    # plot
    filename = f"{bias_flag}_{sent_flag}_p{perplexity}"
    plot_tsne(
        word_vectors=all_word_vectors, perplexity=perplexity, filename=filename, do_tsne=do_tsne,
    )


if __name__ == "__main__":
    seed = random.randint(100, 1000000)
    seed = 17207
    print(f"Set seed: {seed}")
    set_seed(seed)

    if "word2sents6.jsonl" not in os.listdir(DATA_DIR):
        # make data first
        match()

    # tsne only
    for perplexity in [4]:
        visualize_few_words(debiased=True, do_tsne=True, perplexity=perplexity, use_sents=True, model_name="bert")
