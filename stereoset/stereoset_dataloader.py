import json
import string
from typing import List, Union
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer


class IntrasentenceLoader:
    def __init__(
        self,
        tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
        max_seq_len: int,
        pad_to_max_len: bool,
        test_data_path: str,
        model_name_or_path: str,
    ):
        stereoset = StereoSet(test_data_path)
        clusters = stereoset.get_intrasentence_examples()
        self.tokenizer = tokenizer
        self.sentences = []
        self.max_seq_len = max_seq_len
        self.pad_to_max_len = pad_to_max_len
        self.model_name_or_path = model_name_or_path

        for cluster in clusters:
            for sentence in cluster.sentences:
                if self.model_name_or_path is not None and self.model_name_or_path == "roberta-base":
                    insertion_tokens = self.tokenizer.encode(text=f"{sentence.template_word}", add_special_tokens=False)
                    target_tokens = self.tokenizer.encode(text=f"{cluster.target}", add_special_tokens=False)
                else:
                    insertion_tokens = self.tokenizer.encode(text=sentence.template_word, add_special_tokens=False)
                    target_tokens = self.tokenizer.encode(text=cluster.target, add_special_tokens=False)

                for idx in range(len(insertion_tokens)):
                    insertion = self.tokenizer.decode(insertion_tokens[:idx])
                    insertion_string = f"{insertion}{self.tokenizer.mask_token}"
                    new_sentence = cluster.context.replace("BLANK", insertion_string)
                    next_token = insertion_tokens[idx]
                    self.sentences.append((new_sentence, sentence.id, next_token, target_tokens))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence, sentence_id, next_token, target_tokens = self.sentences[idx]
        text = sentence
        text_pair = None
        tokens_dict = self.tokenizer.encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
            pad_to_max_length=self.pad_to_max_len,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
        )
        input_ids = tokens_dict["input_ids"]
        attention_mask = tokens_dict["attention_mask"]
        token_type_ids = tokens_dict["token_type_ids"]

        return sentence_id, next_token, input_ids, attention_mask, token_type_ids, target_tokens


class StereoSet:
    def __init__(self, test_data_path: str):
        with open(file=test_data_path, mode="r") as fp:
            self.json_file = json.load(fp)

        self.version = self.json_file["version"]
        self.intrasentence_examples = self._create_intrasentence_examples(
            examples=self.json_file["data"]["intrasentence"]
        )

    def _create_intrasentence_examples(self, examples: List[dict]):
        created_examples = []
        for example in examples:
            sentences = []
            for sentence in example["sentences"]:
                labels = []
                for label in sentence["labels"]:
                    labels.append(Label(**label))
                sentence_obj = Sentence(
                    id=sentence["id"], sentence=sentence["sentence"], labels=labels, gold_label=sentence["gold_label"]
                )
                word_idx = None
                for idx, word in enumerate(str.split(example["context"], " ")):
                    if "BLANK" in word:
                        word_idx = idx

                template_word = str.split(sentence["sentence"], " ")[word_idx]
                sentence_obj.template_word = str.translate(template_word, str.maketrans("", "", string.punctuation))
                sentences.append(sentence_obj)

            created_example = IntrasentenceExample(
                id=example["id"],
                bias_type=example["bias_type"],
                target=example["target"],
                context=example["context"],
                sentences=sentences,
            )
            created_examples.append(created_example)

        return created_examples

    def get_intrasentence_examples(self):
        return self.intrasentence_examples


class Label:
    def __init__(self, human_id: str, label: str):
        self.human_id = human_id
        self.label = label


class Sentence:
    def __init__(self, id: str, sentence: str, labels: List[str], gold_label: str):
        self.id = id
        self.sentence = sentence
        self.labels = labels
        self.gold_label = gold_label
        self.template_word = None

    def __str__(self) -> str:
        return f"{self.gold_label.capitalize()} Sentence: {self.sentence}"


class Example:
    def __init__(self, id: str, bias_type: str, target: str, context: str, sentences: List[str]):
        self.id = id
        self.bias_type = bias_type
        self.target = target
        self.context = context
        self.sentences = sentences

    def __str__(self) -> str:
        s = f"Domain: {self.bias_type} - Target: {self.target} \r\n"
        s += f"Context: {self.context} \r\n"
        for sentence in self.sentences:
            s += f"{sentence} \r\n"

        return s


class IntrasentenceExample(Example):
    def __init__(self, id: str, bias_type: str, target: str, context: str, sentences: List[str]):
        super().__init__(id, bias_type, target, context, sentences)
