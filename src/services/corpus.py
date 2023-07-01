# File: corpus.py, Project: SpokenNer
# Created by Moncef Benaicha
# Contact: support@moncefbenaicha.me
import json

from src.services import logger

import re
import torchaudio
from typing import Dict, List, Union
from dataclasses import dataclass
from torch.utils.data import Dataset
import torch
from transformers import Wav2Vec2Processor

resampler = {
    32000: torchaudio.transforms.Resample(orig_freq=32000, new_freq=16000),
    44100: torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000),
    48000: torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000),
}


@dataclass
class Sample:
    tokens: List[str]
    labels: List[str]
    text: str
    audio_path: str
    target_text: str
    processed_text: str = None
    predicted_text: str = None
    predicted_augmented_text: str = None
    predicted_tokens: List[str] = None
    predicted_labels: List[str] = None

    def as_dict(self):
        return {
            "tokens": self.tokens,
            "labels": self.labels,
            "text": self.text,
            "audio_path": self.audio_path,
            "target_text": self.target_text,
            "processed_text": self.processed_text,
            "predicted_text": self.predicted_text,
            "predicted_augmented_text": self.predicted_augmented_text,
            "predicted_tokens": self.predicted_tokens,
            "predicted_labels": self.predicted_labels,
        }


@dataclass
class Span:

    start_idx: int
    end_idx: int
    type: str

    def get_labels(self):
        labels = [f'B-{self.type}'] + [f'I-{self.type}'] * (self.end_idx-self.start_idx-1)
        return self.start_idx, self.end_idx, labels


@dataclass
class SpansBuilder:

    start_idx: int = None
    type: str = None
    spans: list = None

    def __post_init__(self):
        self.spans = []

    def new_span(self, start_idx: int, entity_type: str):

        self.start_idx = start_idx
        self.type = entity_type

    def end_span(self, end_idx: int):
        if self.start_idx is not None:
            self.spans.append(
                Span(self.start_idx, end_idx, self.type)
            )
        self.start_idx = None
        self.type = None

    def rematch_labels(self, labels):
        for span in self.spans:
            start, end, tmp_labels = span.get_labels()
            labels[start:end] = tmp_labels


@dataclass
class DataPreProcessor:
    END_TAG = "]"

    NER_TAGS = {
        "ORG": "{",
        "PER": "[",
        "LOC": "$"
    }

    REVERSE_NER_TAGS = {v: k for k, v in NER_TAGS.items()}

    lowercase: bool = True

    def ner_transcript_tagging(self, tokens: List[str], labels: List[str]):
        output_sentence = ""
        for j, (token, tag) in enumerate(zip(tokens, labels)):
            if tag != "O":
                tag = tag.split("-")
                if tag[1] in self.NER_TAGS.keys():
                    if tag[0] == "B":
                        token = f"{self.NER_TAGS[tag[1]]} {token}"
                    if (j < len(labels) - 1 and (
                            labels[j + 1] == "O" or labels[j + 1].split("-")[0] == "B")) or (j == len(labels) - 1):
                        "Case of the tag inside, check if the next is O then end tag otherwise do nothing"
                        token = f"{token} {self.END_TAG}"
            output_sentence += f"{token} "
        return re.sub(r"\s+", " ", output_sentence.strip())

    def reverse_ner_transcript_tagging(self, transcript: str):
        builder = SpansBuilder()
        predicted_text = re.sub(r'\s+', ' ', re.sub(r"[\{\[\$\]]", "", transcript)).strip()

        # Remove any additional space between flags ie: $ Mars ] ==> $Mars]
        # Remove any standalone tag ie: I was in $ Boston ] [ ==> I was in $Boston]
        # Remove empty tagging ie: I was {] in $ Boston ] ==> I was in $Boston]

        transcript = re.sub(r'(\{|\$|\[)\s+', r"\g<1>", transcript)
        transcript = re.sub(r'\s+\]', ']', transcript)
        transcript = re.sub(r'(\{|\[|\$|\])(\{|\[|\$|\])', '', transcript)

        tokens = predicted_text.split()
        labels = ['O'] * len(tokens)
        i = 0
        for token in transcript.split():
            if not re.search(r'\w+', token):
                continue
            if token[0] in self.REVERSE_NER_TAGS.keys():
                builder.new_span(i, self.REVERSE_NER_TAGS[token[0]])
            if token.endswith("]"):
                builder.end_span(i + 1)
            i += 1
        builder.rematch_labels(labels)
        return tokens, labels, predicted_text

    def pre_process_utterance(self, utterance: Sample):
        new_tokens = []
        new_labels = []
        for token, label in zip(utterance.tokens, utterance.labels):
            token = token.lower().strip()
            if token:
                new_tokens.append(token)
                new_labels.append(label)
        utterance.tokens = new_tokens
        utterance.labels = new_labels
        utterance.processed_text = self.ner_transcript_tagging(utterance.tokens, utterance.labels)


class Corpus(Dataset):

    def __init__(
            self,
            dataset: list,
            processor: Wav2Vec2Processor,
            lowercase: bool = True,
            clear_punctuation: bool = True,
            evaluate_mode: bool = False
    ):
        super(Corpus).__init__()

        self.wav2vec_processor = processor
        self.data_processor = DataPreProcessor()
        self.dataset: List[Sample] = self.reformat_data(dataset)
        for utterance in self.dataset:
            self.data_processor.pre_process_utterance(utterance)
        self.evaluate_mode = evaluate_mode
        logger.info(
            f"Dataset size: {len(self.dataset)}. "
            f"Lower case: {lowercase}, Remove punctuation: {clear_punctuation}"
        )
        logger.info(f"Utterance input example: \n\n{json.dumps(self.dataset[6].as_dict(), indent=4)}\n\n")

    def reformat_data(self, dataset):
        output = []
        for utterance in dataset:
            output.append(
                Sample(**{
                    "tokens": utterance[0],
                    "labels": utterance[1],
                    "text": utterance[2],
                    "target_text": utterance[2].lower().strip(),
                    "audio_path": utterance[3],
                })
            )
        return output

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        audio, sr = torchaudio.load(sample.audio_path)
        if sr != 16000:
            audio = resampler[sr].forward(audio.squeeze(0)).numpy()
        else:
            audio = audio.squeeze(0).numpy()

        if self.evaluate_mode:
            return audio, index

        out = self.wav2vec_processor(
            audio=audio,
            sampling_rate=16000,
            text=sample.processed_text
        )

        return {
            "input_values": out["input_values"][0],
            "attention_mask": out["attention_mask"][0],
            "labels": out["labels"],
        }

    def get_dataset(self):
        return [
            x.as_dict() for x in self.dataset
        ]


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        # Notation Mismatch between data_loader and wav2Vec.
        # Wav2Vec expect input_values instead dataloader expect input_ids
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features=input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        batch["labels"] = labels_batch.input_ids.masked_fill(labels_batch.attention_mask.ne(1), -100)

        return batch
