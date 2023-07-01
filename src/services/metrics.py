# File: metrics.py, Project: SpokenNer
# Created by Moncef Benaicha
# Contact: support@moncefbenaicha.me

from src.services import logger

from dataclasses import dataclass
from transformers import (
    Wav2Vec2Processor
)
from seqscore.encoding import get_encoding
from seqscore.model import LabeledSequence
from seqscore.scoring import compute_scores
from evaluate import (
    load,
    EvaluationModule
)
import numpy as np
from typing import List
from src.services.corpus import Sample


seqscore_encoder = get_encoding("BIO")


class EntityErrorRate:

    def __init__(self):
        self.total_entities = 0
        self.correct_entities = 0
        self.incorrect_entities = 0

    def __call__(self, predicted_transcript: str, reference_entities: LabeledSequence, *args, **kwargs):
        for mention in reference_entities.mentions:
            tokens_seq = ' '.join(reference_entities.mention_tokens(mention))
            if tokens_seq in predicted_transcript:
                self.correct_entities += 1
            else:
                self.incorrect_entities += 1
            self.total_entities += 1

    @property
    def entity_error_rate(self):
        return round(self.incorrect_entities/self.total_entities, 2)


@dataclass
class Metrics:
    processor: Wav2Vec2Processor
    wer: EvaluationModule = load("wer")
    cer: EvaluationModule = load("cer")
    entity_metric = EntityErrorRate()

    def evaluate_pred(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer_score = self.wer.compute(predictions=pred_str, references=label_str)
        cer_score = self.cer.compute(predictions=pred_str, references=label_str)

        return {"wer": wer_score, "cer": cer_score}

    def report_metrics(self, dataset: List[Sample]):
        original_transcript = []
        predicted_transcript = []
        predicted_sequences = []
        original_sequences = []
        for i, sample in enumerate(dataset):
            original_transcript.append(sample.target_text)
            predicted_transcript.append(sample.predicted_text)
            if len(sample.predicted_tokens) and len(sample.predicted_labels):
                pred_sequence = LabeledSequence(
                        tokens=tuple(sample.predicted_tokens),
                        labels=tuple(sample.predicted_labels),
                        mentions=tuple(
                            seqscore_encoder.decode_labels(sample.predicted_labels, sample.predicted_tokens)),
                    )
                ref_sequence = LabeledSequence(
                        tokens=tuple(sample.tokens),
                        labels=tuple(sample.labels),
                        mentions=tuple(seqscore_encoder.decode_labels(sample.labels, sample.tokens)),
                    )
                predicted_sequences.append(pred_sequence)
                original_sequences.append(ref_sequence)
                self.entity_metric(sample.predicted_text, ref_sequence)
            else:
                logger.error(
                    f"Something wrong with sentence id: {i},"
                    f" Token: {sample.predicted_tokens}, Labels: {sample.predicted_labels}")
        class_scores, acc_scores = compute_scores([predicted_sequences], [original_sequences])
        meta = {
            "asr": {
                "wer": self.wer.compute(predictions=predicted_transcript, references=original_transcript)*100,
                "cer": self.cer.compute(predictions=predicted_transcript, references=original_transcript)*100,
                "entity_er": self.entity_metric.entity_error_rate*100,
            },
            "ner": {
                "accuracy": acc_scores.accuracy,
                "f1": class_scores.f1,
                "tags": {
                    k: {
                        "f1": class_scores.type_scores[k].f1,
                    } for k in class_scores.type_scores.keys() if k != "default_factory"
                }
            }
        }
        return meta




