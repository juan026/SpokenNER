#%%


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
import torchaudio
import json
import torch
import argparse
import os
import random
import time
from transformers import Wav2Vec2Processor
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
import spacy
from spacy import displacy

resampler = {
    32000: torchaudio.transforms.Resample(orig_freq=32000, new_freq=16000),
    44100: torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000),
    48000: torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000),
}

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


def evaluate(data_path, device_name, device, asr_model, vocab_path):
    logger.info("************* Evaluate function **************")
    logger.info(f"Using Device: {device} {'name: ' + device_name if device != 'cpu' else ''}")
    logger.info(f"Loading model from: {asr_model}")
    asr = ASR.load_for_evaluation(asr_model, vocab_path)
    asr.model.to(device)
    asr.model.eval()
   
    # **************** Data Loading ******************** #
    logger.info(
        f"Loading test set from provided source: {data_path}"
    )
    pre_processor = DataPreProcessor()

    spacy_ner = spacy.load("en_core_web_sm")



  
   
    logger.info("Evaluation started ...")
    for f in data_path:
        file_name, target_transcription = f

        audio, sr = torchaudio.load(file_name)#os.path.join(clips_path, file_name))
        if sr != 16000:
            audio = resampler[sr].forward(audio.squeeze(0)).numpy()
        else:
            audio = audio.squeeze(0).numpy()

        
        out = asr.processor(
            audio=audio,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )

        features= {
            "input_values": out["input_values"],
            "attention_mask": out["attention_mask"]
        }


        
        with torch.no_grad():
            output = asr.model(
                input_values=features['input_values'].to(device),
                attention_mask=features['attention_mask'].to(device)
            )
        
        prediction = asr.processor.batch_decode(torch.argmax(output.logits, dim=-1))
        tokens, labels, predicted_text = pre_processor.reverse_ner_transcript_tagging(prediction[0])

        print("Original Text: {}".format(target_transcription))
        print("Predicted Text: {}".format(predicted_text))
        print()
        print("###################################################")
        print("Spacy: ")
        output_spacy_ner = spacy_ner(predicted_text)
        for word in output_spacy_ner.ents:
            print(word.text, word.label_)
        #displacy.render(output_spacy_ner, style="ent", jupyter=True)
        print("###################################################")
        print()
        print("###################################################")
        print("Spoken NER:")
        for t, l in zip(tokens, labels):
            print("Token: {} - Label: {}".format(t, l))
        print("###################################################")
        print()
   
    logger.info("""
        Done.
    """)
    src.services.send_notification("Evaluation Finished")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='Spoken NER',
                    description='Run inferences on mp3 audio recordings.',)
    parser.add_argument('--asr_model', type=str, default="/home/juan/NLP_Project1/model_trained") 
    parser.add_argument('--audio_path', type=str, default="/home/juan/NLP_Project1/Alex.mp3") 
    parser.add_argument('--audio_transcription', type=str, default="")
    parser.add_argument('--vocab_path', type=str, default="/home/juan/NLP_Project1/cv-corpus-12.0-2022-12-07-en/en/en_vocab_with_tags.json") 
    parser.add_argument('--output_path', type=str, default="/home/juan/NLP_Project1/cv-corpus-12.0-2022-12-07-en/evalOffline/") 

    args = parser.parse_args()

   
    # ***************** Evaluate *****************

    #Path of the ASR model
    asr_model = args.asr_model
        
       
    task = "End2end Evaluation en XLS-R 300" 
        
 
    data_path = [[args.audio_path, args.audio_transcription]]
    #["common_voice_en_27710027.mp3", "Joe Keaton disapproved of films and Buster also had reservations about the medium"]]
    
       

    vocab_path = args.vocab_path
       
    seed = 77773
  
    output_path = args.output_path
       
  
    init(seed)

    device, device_name = set_device(False)

    src.services.task = task
    
    os.makedirs(output_path, exist_ok=True)
    log_factory.change_log_output_file(
        logger, os.path.join(output_path, "output.log")
    )
    evaluate(data_path, device_name, device, asr_model, vocab_path)
