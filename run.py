# File: run.py, Project: SpokenNer
# Created by Moncef Benaicha
# Contact: support@moncefbenaicha.me

import src.services
from src.services import log_factory
from src.services import logger
from src.services.tools import (
    init,
    set_device,
    correct_audio_path,
    data_portion_selection
)
from src.services.corpus import Corpus, DataCollatorCTCWithPadding, DataPreProcessor
from src.services.data_loader import Json
from src.models.Wav2Vec import ASR
from src.services.metrics import Metrics

import json
import torch
import argparse
import os
import random
import time
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    PrinterCallback
)
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class LoggingCallback(TrainerCallback):

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info("Training Started")

    def on_log(self, args, state, control, logs=None, **kwargs):
        current = state.log_history[-1]
        if 'eval_loss' in current.keys():
            "Validation Step"
            logger.info(
                f'Validation, Epoch: {int(state.epoch) + 1}/{state.num_train_epochs}, Step: {state.global_step}/{state.max_steps}, Validation Loss: {current["eval_loss"]:.4f}, WER: {current["eval_wer"]*100:.2f}%, CER: {current["eval_cer"]*100:.2f}%, Evaluation Runtime: {current["eval_runtime"]:.3f}s')
        elif 'loss' in current:
            "Training Step"
            logger.info(
                f'Training, Epoch: {int(state.epoch)+1}/{state.num_train_epochs}, Step: {state.global_step}/{state.max_steps}, Training Loss: {current["loss"]:.4f}, Lr: {current["learning_rate"]:.9f}')


def evaluate():
    logger.info("************* Evaluate function **************")
    logger.info(f"Using Device: {device} {'name: ' + device_name if device != 'cpu' else ''}")
    logger.info(f"Loading model from: {args.asr_model}")
    asr = ASR.load_for_evaluation(args.asr_model, args.vocab_path, args.language_model)
    asr.model.to(device)
    asr.model.eval()
    if args.language_model:
        logger.info(f"LM Hyper-Parameters: Alpha={args.lm_alpha} Beta={args.lm_beta} Beam-Size={args.lm_beam_size}")

    # **************** Data Loading ******************** #
    logger.info(
        f"Loading test set from provided source: {args.data_path}"
    )
    data = Json.load(args.data_path)
    test_set = data["test"]
    del data
    correct_audio_path(test_set, args.clips_path)
    t1 = time.time()
    test_corpus = Corpus(
        dataset=test_set,
        processor=asr.processor,
        evaluate_mode=True
    )
    logger.info(f"Test set processing finished, elapsed time: {time.time() - t1:.2f}")

    def special_batch(batch):
        audios = []
        samples = []
        for x in batch:
            audios.append(x[0])
            samples.append(x[1])
        features = asr.processor(
            audio=audios,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )
        return features, samples

    data_loader = DataLoader(
        dataset=test_corpus,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=special_batch,
        pin_memory=True
    )
    pre_processor = DataPreProcessor()
    logger.info("Evaluation started ...")
    for batch in tqdm(data_loader):
        features, samples = batch
        with torch.no_grad():
            output = asr.model(
                input_values=features.input_values.to(device),
                attention_mask=features.attention_mask.to(device)
            )
        if args.language_model:
            predicted = asr.processor.batch_decode(
                logits=output.logits.cpu().numpy(),
                beam_width=args.lm_beam_size,
                alpha=args.lm_alpha,
                beta=args.lm_beta
            ).text
        else:
            predicted = asr.processor.batch_decode(torch.argmax(output.logits, dim=-1))
        for (index, prediction) in zip(samples, predicted):
            tokens, labels, predicted_text = pre_processor.reverse_ner_transcript_tagging(prediction)
            test_corpus.dataset[index].predicted_tokens = tokens
            test_corpus.dataset[index].predicted_labels = labels
            test_corpus.dataset[index].predicted_text = predicted_text
            test_corpus.dataset[index].predicted_augmented_text = prediction
    with open(os.path.join(args.output_path, "evaluation.json"), "w") as output_stream:
        json.dump(
            {
                "data": test_corpus.get_dataset()
            }, output_stream, ensure_ascii=False, indent=4)
    logger.info("Calculating Metrics")
    metrics = Metrics(processor=asr.processor)
    meta = metrics.report_metrics(test_corpus.dataset)
    with open(os.path.join(args.output_path, "evaluation.json"), "w") as output_stream:
        json.dump(
            {
                "meta": meta,
                "data": test_corpus.get_dataset()
            }, output_stream, ensure_ascii=False, indent=4)
    logger.info("""
        Done.
    """)
    src.services.send_notification("Evaluation Finished")


def train():
    # **************** MODEL DEFINITION **************** #
    asr = ASR.load_model(
        model_name=args.pretrained_model,
        vocab_path=args.vocab_path
    )
    asr.model.freeze_feature_encoder()
    logger.info(f'ASR Model Size: {sum(p.numel() for p in asr.model.parameters())}')

    # **************** Data Loading ******************** #
    logger.info(
        f"Loading data from provided source: {args.data_path}, with clips path: {args.clips_path}"
    )
    t1 = time.time()
    data = Json.load(args.data_path)
    logger.info(
        f"Data Loaded\n"
        f"Train set: {len(data['train'])} utterances\n"
        f"Valid set: {len(data['valid'])} utterances\n"
        f"Test set: {len(data['test'])} utterances\n"
        f"Elapsed time: {time.time()-t1:.2f}s"
    )
    train_set = data["train"]
    correct_audio_path(train_set, args.clips_path)
    valid_set = data["valid"]
    correct_audio_path(valid_set, args.clips_path)
    del data
    if len(args.multilingual):
        assert len(args.multilingual) == len(args.multilingual_audio), "Two variables must be equal length"
        logger.info("Multilingual Data Loading")
        for path, clips in zip(args.multilingual, args.multilingual_audio):
            logger.info(
                f"Loading data from provided source: {path}, with clips path: {clips}"
            )
            tmp_data = Json.load(path)
            correct_audio_path(tmp_data["train"], clips)
            train_set += tmp_data["train"]
            correct_audio_path(tmp_data["valid"], clips)
            valid_set += tmp_data["valid"]
    logger.info(
        f"Data Loaded\n"
        f"Train set: {len(train_set)} utterances\n"
        f"Valid set: {len(valid_set)} utterances\n"
        f"Elapsed time: {time.time() - t1:.2f}s"
    )
    np.random.shuffle(train_set)
    np.random.shuffle(valid_set)
    t1 = time.time()

    with open(os.path.join(args.output_path, "train_data_snapshot.json"), "w") as output_stream:
        json.dump({
            "data": train_set
        }, output_stream, indent=4, ensure_ascii=False)

    train_corpus = Corpus(
        dataset=train_set,
        processor=asr.processor
    )
    logger.info(f"Train set processing finished, elapsed time: {time.time()-t1:.2f}")
    t1 = time.time()
    valid_corpus = Corpus(
        dataset=valid_set,
        processor=asr.processor
    )
    logger.info(f"Valid set processing finished, elapsed time: {time.time() - t1:.2f}")

    # **************** TRAINER ************************ #
    logger.info(f"Number of epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")

    steps = int(np.ceil(len(train_corpus) / args.batch_size) * args.epochs / torch.cuda.device_count())
    warmup = int(steps / 3)
    save_eval_steps = 5000
    logger.info(f"Number of steps: {steps}")
    logger.info(f"Warmup steps: {warmup}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Evaluation and Save steps: {save_eval_steps}")
    logger.info(f"Device set to: {device}{', '+device_name if device!= 'cpu' else ''}")

    metrics = Metrics(processor=asr.processor)
    data_collator = DataCollatorCTCWithPadding(processor=asr.processor)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        group_by_length=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=save_eval_steps,
        eval_steps=save_eval_steps,
        eval_accumulation_steps=20,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        ignore_data_skip=True,
        logging_steps=10,
        learning_rate=args.lr,
        warmup_steps=warmup,
        save_total_limit=2,
        log_level="info",
        logging_dir=os.path.join(args.output_path, 'logs'),
        optim="adamw_torch",
        push_to_hub=False,
        disable_tqdm=True,
        load_best_model_at_end=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        seed=args.seed
    )

    trainer = Trainer(
        model=asr.model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=metrics.evaluate_pred,
        train_dataset=train_corpus,
        eval_dataset=valid_corpus,
        tokenizer=asr.processor.tokenizer,
        callbacks=[LoggingCallback]
    )
    trainer.remove_callback(PrinterCallback)
    t1 = time.time()
    src.services.send_notification("Training Started")
    if args.checkpoint:
        trainer.train(os.path.join(args.output_path, args.checkpoint))
    else:
        trainer.train()
    logger.info(f"Training finished. Elapsed time: {(time.time()-t1)/3600:.2f}h")
    src.services.send_notification("Training Finished")
    trainer.save_state()
    with open(os.path.join(args.output_path, "training_args.json"), "w") as output_stream:
        json.dump(training_args.to_dict(), output_stream, ensure_ascii=False, indent=4)


def transfer():
    # **************** MODEL DEFINITION **************** #
    asr = ASR.load_model(
        model_name=args.source_model,
        vocab_path=args.vocab_path
    )
    asr.model.freeze_feature_encoder()
    logger.info(f'ASR Model Size: {sum(p.numel() for p in asr.model.parameters())}')

    # **************** Data Loading ******************** #
    logger.info(
        f"Loading data from provided source: {args.data_path}, with clips path: {args.clips_path}"
    )
    t1 = time.time()
    data = Json.load(args.data_path)
    logger.info(
        f"Data Loaded\n"
        f"Train set: {len(data['train'])} utterances\n"
        f"Valid set: {len(data['valid'])} utterances\n"
        f"Test set: {len(data['test'])} utterances\n"
        f"Elapsed time: {time.time() - t1:.2f}s"
    )
    train_set = data_portion_selection(data["train"], args.data_proportion)
    correct_audio_path(train_set, args.clips_path)
    valid_set = data["valid"]
    correct_audio_path(valid_set, args.clips_path)
    del data
    np.random.shuffle(train_set)
    np.random.shuffle(valid_set)
    t1 = time.time()

    with open(os.path.join(args.output_path, "train_data_snapshot.json"), "w") as output_stream:
        json.dump({
            "data": train_set
        }, output_stream, indent=4, ensure_ascii=False)

    train_corpus = Corpus(
        dataset=train_set,
        processor=asr.processor
    )
    logger.info(f"Train set processing finished, elapsed time: {time.time() - t1:.2f}")
    t1 = time.time()
    valid_corpus = Corpus(
        dataset=valid_set,
        processor=asr.processor
    )
    logger.info(f"Valid set processing finished, elapsed time: {time.time() - t1:.2f}")

    # **************** TRAINER ************************ #
    logger.info(f"Number of epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")

    steps = int(np.ceil(len(train_corpus) / args.batch_size) * args.epochs / torch.cuda.device_count())
    warmup = int(steps / 3)
    save_eval_steps = 200
    logger.info(f"Number of steps: {steps}")
    logger.info(f"Warmup steps: {warmup}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Evaluation and Save steps: {save_eval_steps}")
    logger.info(f"Device set to: {device}{', ' + device_name if device != 'cpu' else ''}")

    metrics = Metrics(processor=asr.processor)
    data_collator = DataCollatorCTCWithPadding(processor=asr.processor)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        group_by_length=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=save_eval_steps,
        eval_steps=save_eval_steps,
        eval_accumulation_steps=20,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        ignore_data_skip=True,
        logging_steps=10,
        learning_rate=args.lr,
        warmup_steps=warmup,
        save_total_limit=2,
        log_level="info",
        logging_dir=os.path.join(args.output_path, 'logs'),
        optim="adamw_torch",
        push_to_hub=False,
        disable_tqdm=True,
        load_best_model_at_end=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        seed=args.seed
    )

    trainer = Trainer(
        model=asr.model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=metrics.evaluate_pred,
        train_dataset=train_corpus,
        eval_dataset=valid_corpus,
        tokenizer=asr.processor.tokenizer,
        callbacks=[LoggingCallback]
    )
    trainer.remove_callback(PrinterCallback)
    t1 = time.time()
    src.services.send_notification("Training Started")
    trainer.train()
    logger.info(f"Training finished. Elapsed time: {(time.time() - t1) / 3600:.2f}h")
    src.services.send_notification("Training Finished")
    trainer.save_state()
    with open(os.path.join(args.output_path, "training_args.json"), "w") as output_stream:
        json.dump(training_args.to_dict(), output_stream, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    sub_parser = parser.add_subparsers(
        dest="command",
        help="What should the script run?",
        required=True,
    )

    # ***************** Training *****************

    train_parser = sub_parser.add_parser(
        "train", help="End2End Model training"
    )

    train_parser.add_argument(
        "--pretrained_model",
        help="Pretrained model to use for training",
        required=True,
    )

    train_parser.add_argument(
        "--multilingual",
        help="data paths for other languages",
        required=False,
        default=[],
        nargs="*"
    )

    train_parser.add_argument(
        "--multilingual_audio",
        help="audio paths for other languages",
        required=False,
        default=[],
        nargs="*"
    )

    train_parser.add_argument(
        "--epochs",
        help="Number of training epochs",
        type=int,
        default=10
    )

    train_parser.add_argument(
        "--lr",
        help="Value for learning rate",
        required=False,
        type=float,
        default=1e-4
    )

    train_parser.add_argument(
        "--checkpoint",
        help="String of the checkpoint to use, if available",
        required=False,
        type=str
    )

    # ***************** Evaluate *****************

    evaluate_parser = sub_parser.add_parser(
        "evaluate"
    )
    evaluate_parser.add_argument(
        "--asr_model",
        help="Path of the ASR model",
        required=True
    )

    evaluate_parser.add_argument(
        "--language_model",
        help="Path of the Language Model",
        required=False
    )

    evaluate_parser.add_argument(
        "--lm_alpha",
        type=float,
        required=False
    )

    evaluate_parser.add_argument(
        "--lm_beta",
        type=float,
        required=False
    )

    evaluate_parser.add_argument(
        "--lm_beam_size",
        type=int,
        required=False
    )

    # ***************** Transfer *****************

    transfer_parser = sub_parser.add_parser(
        "transfer", help="End2End Model transfer learning"
    )

    transfer_parser.add_argument(
        "--source_model",
        help="Path to end-to-end model",
        required=True,
    )

    transfer_parser.add_argument(
        "--data_proportion",
        help="Amount of data to consider for training",
        required=True,
        default=20,
        type=int
    )

    transfer_parser.add_argument(
        "--epochs",
        help="Number of training epochs",
        type=int,
        default=10
    )

    transfer_parser.add_argument(
        "--lr",
        help="Value for learning rate",
        required=False,
        type=float,
        default=1e-4
    )

    # ***************** General *****************
    parser.add_argument(
        "--task",
        help="Task Description",
        required=True
    )

    parser.add_argument(
        "--data_path",
        help="Path for JSON Data source that include test set",
        required=True
    )

    parser.add_argument(
        "--clips_path",
        help="Path to folder that contains audio files",
        required=True
    )

    parser.add_argument(
        "--vocab_path",
        help="Path to folder that contains vocabulary",
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=48
    )
    parser.add_argument(
        "--gpu",
        help="Flag to activate the use of GPU",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--lowercase",
        help="Flag to activate string transformation to lower case",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--clear_punctuation",
        help="Flag to activate string punctuation removal",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--seed", help="Seed number to be used in randomization", type=int, default=77773
    )
    parser.add_argument(
        "--output_path",
        help="Directory for output files",
        required=True
    )

    args = parser.parse_args()

    if not (seed := args.seed):
        seed = random.randint(1000, 9999)

    init(seed)

    device, device_name = set_device(args.gpu)

    src.services.task = args.task
    match args.command:
        case "train":
            os.makedirs(args.output_path, exist_ok=True)
            log_factory.change_log_output_file(
                logger, os.path.join(args.output_path, "output.log")
            )
            train()
        case "evaluate":
            os.makedirs(args.output_path, exist_ok=True)
            log_factory.change_log_output_file(
                logger, os.path.join(args.output_path, "output.log")
            )
            evaluate()
        case "transfer":
            os.makedirs(args.output_path, exist_ok=True)
            log_factory.change_log_output_file(
                logger, os.path.join(args.output_path, "output.log")
            )
            transfer()
        case _:
            logger.error("Requested command not found")
