# File: Wav2Vec.py, Project: SpokenNer
# Created by Moncef Benaicha
# Contact: support@moncefbenaicha.me

from src.services import logger
from dataclasses import dataclass
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ProcessorWithLM
)
from transformers.processing_utils import ProcessorMixin
from pyctcdecode import build_ctcdecoder

unk_token = "<unk>"
pad_token = "<pad>"
word_delimiter_token = "|"


@dataclass
class ASR:
    model: Wav2Vec2ForCTC
    processor: ProcessorMixin

    @classmethod
    def load_model(cls, model_name, vocab_path):
        processor = Wav2Vec2Processor(
                feature_extractor=Wav2Vec2FeatureExtractor(
                    feature_size=1,
                    sampling_rate=16000,
                    padding_value=0.0,
                    do_normalize=True,
                    return_attention_mask=True
                ),
                tokenizer=Wav2Vec2CTCTokenizer(
                    vocab_path,
                    unk_token=unk_token,
                    pad_token=pad_token,
                    word_delimiter_token=word_delimiter_token
                )
            )
        logger.info(f"Loading model files from: {model_name}")
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
        return cls(
            model=model,
            processor=processor
        )

    @classmethod
    def load_for_evaluation(cls, model_path, vocab_path, language_model=None):
        processor = Wav2Vec2Processor(
            feature_extractor=Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True
            ),
            tokenizer=Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=unk_token,
                pad_token=pad_token,
                word_delimiter_token=word_delimiter_token
            )
        )
        if language_model:
            logger.info(f"Loading Language model from {language_model}")
            decoder = build_ctcdecoder(
                labels=[
                    k for k, _ in sorted(processor.tokenizer.get_vocab().items(), key=lambda item: item[1])
                ],
                kenlm_model_path=language_model
            )
            processor = Wav2Vec2ProcessorWithLM(
                feature_extractor=processor.feature_extractor,
                tokenizer=processor.tokenizer,
                decoder=decoder
            )
        logger.info(f"Loading model files from: {model_path}")
        return cls(
            model=Wav2Vec2ForCTC.from_pretrained(model_path),
            processor=processor
        )

