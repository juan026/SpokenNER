<div style="text-align:center">

  <h3>Exploring Spoken Named Entity Recognition: A Cross-Lingual Perspective</h3>
    
  <p>
    Git Repo for the paper source code and data
    <br />
    <a href="">Paper</a>
    ·
    <a href="https://doi.org/10.5281/zenodo.8104278">Data</a>
    ·
    <a href="mailto:support@moncefbenaicha.me">Contact</a>
  </p>
    <br />
    <br />
    <br />
</div>


<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#paper-abstract">Paper Abstract</a></li>
    <li><a href="#about-this-repo">About This Repo</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#usage">Results</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#paper-citation">Paper Citation</a></li>
  </ol>
</details>


## Paper Abstract
Recent advancements in Named Entity Recognition (NER) have significantly improved the identification of entities in textual data. However, spoken NER, a specialized field of spoken document retrieval, lags behind due to its limited research and scarce datasets. Moreover, cross-lingual transfer learning in spoken NER has remained unexplored. This paper utilizes transfer learning across Dutch, English, and German using pipeline and End-to-End (E2E) schemes. We employ Wav2Vec2-XLS-R models on custom pseudo-annotated datasets and investigate several architectures for the adaptability of cross-lingual systems. Our results demonstrate that End-to-End spoken NER outperforms pipeline-based alternatives over our limited annotations. Notably, transfer learning from German to Dutch surpasses the Dutch E2E system by 7% and the Dutch pipeline system by 4%. This study not only underscores the feasibility of transfer learning in spoken NER but also sets promising outcomes for future evaluations, hinting at the need for comprehensive data collection to augment the results.

## About This Repo
This repository includes the following:
* The source code we used to run different experiments in this work
* Links to access different datasets and data format description
* Usage guide to re-run the experiments

## Data
* For all experiments in this work, we relied on data from Common Voice Corpus v12.0 (07/12/2022) (https://commonvoice.mozilla.org/en/datasets). 
* For every language (English, German, Dutch), we apply some data pre-processing as follows:
  * Remove all duplicated utterances
  * Remove all utterances where the audio standard deviation is equal to or lower than the threshold 0.001 (Empty or very inaudible audio)
  * Remove all punctuation from the transcription, except for English, we keep the apostrophe (').
  * In some utterances, we find that the transcription has multiple languages, for example English sentence with Russian or Chinese names. Those names will be converted to the Latin alphabet. This doesn't apply to special German letters (ä, ö, ü, ß).
* After pre-processing we use our best NER model to generate pseudo-annotation from the transcription, then all the transcription is converted to **lowercase**
* We save the previous steps output of each language on a Json file in the following format:
    <pre>
        {
            "meta": {
                "Contains meta information about the dataset"
            },
            "data": [
                [
                    ["token_1","token_2","token_3",...,"token_n"],
                    ["label_1","label_2","label_3",...,"label_n"],
                    "Original Transcription",
                    "file_name.mp3",
                    "sampling_rate",
                    "audio_length_in_seconds"
                ],
                ...
            ]
        }
    </pre>
* Our source code heavily depends on this format.
* **The Json files can be found on: https://doi.org/10.5281/zenodo.8104278**

## Usage

### Training

<pre>
#!/bin/bash

DATA_BASE_PATH="/data/cv-corpus-12.0-2022-12-07"
MODEL="facebook/wav2vec2-xls-r-300m"
OUTPUT_BASE_PATH="/data/output"
LANGUAGE="en"

python run.py \
  --task "End2end Training ${LANGUAGE} XLS-R 300" \
  --data_path $DATA_BASE_PATH"/${LANGUAGE}/cv_${LANGUAGE}_dataset.json" \
  --vocab_path $DATA_BASE_PATH"/${LANGUAGE}/${LANGUAGE}_vocab_with_tags.json" \
  --clips_path $DATA_BASE_PATH"/${LANGUAGE}/clips" \
  --output_path $OUTPUT_BASE_PATH"/${LANGUAGE}/" \
  train \
    --pretrained_model $MODEL
</pre>

### Transfer Learning

<pre>
#!/bin/bash

DATA_BASE_PATH="/data/cv-corpus-12.0-2022-12-07"
MODEL="/data/output/en/my_best_model_checkpoint"
OUTPUT_BASE_PATH="/data/output/de/"
PORTION=40        # This refers to 40% of target language train data

LANGUAGE="nl"

python run.py \
  --task "Transfer From German to NL XLS-R 300" \
  --data_path $DATA_BASE_PATH"/${LANGUAGE}/cv_${LANGUAGE}_dataset.json" \
  --vocab_path $DATA_BASE_PATH"/de/de_vocab_with_tags.json" \
  --clips_path $DATA_BASE_PATH"/${LANGUAGE}/clips" \
  --output_path $OUTPUT_BASE_PATH"/transfer_to_${LANGUAGE}_p${PORTION}/" \
  transfer \
    --source_model $MODEL \
    --data_proportion ${PORTION} \
    --epochs 30
</pre>

### Evaluation with and without a Language Model

<pre>
#!/bin/bash

DATA_BASE_PATH="/data/cv-corpus-12.0-2022-12-07"
MODEL="/data/output/en/my_best_model_checkpoint"
OUTPUT_BASE_PATH="/data/output"
LANGUAGE="en"

python run.py \
 --task "End2end Evaluation ${LANGUAGE} XLS-R 300" \
 --data_path $DATA_BASE_PATH"/${LANGUAGE}/cv_${LANGUAGE}_dataset.json" \
 --vocab_path $DATA_BASE_PATH"/${LANGUAGE}/${LANGUAGE}_vocab_with_tags.json" \
 --clips_path $DATA_BASE_PATH"/${LANGUAGE}/clips" \
 --output_path $OUTPUT_BASE_PATH"/evaluation/${LANGUAGE}/no_lm" \
 --batch_size 8 \
 evaluate \
   --asr_model $MODEL

python run.py \
 --task "End2end Evaluation ${LANGUAGE} XLS-R 300 with LM" \
 --data_path $DATA_BASE_PATH"/${LANGUAGE}/cv_${LANGUAGE}_dataset.json" \
 --vocab_path $DATA_BASE_PATH"/${LANGUAGE}/${LANGUAGE}_vocab_with_tags.json" \
 --clips_path $DATA_BASE_PATH"/${LANGUAGE}/clips" \
 --output_path $OUTPUT_BASE_PATH"/evaluation/${LANGUAGE}/lm" \
 --batch_size 8 \
 evaluate \
   --asr_model $MODEL \
   --language_model $DATA_BASE_PATH"/${LANGUAGE}/${LANGUAGE}_lm.arpa" \
   --lm_alpha 1.0 \
   --lm_beta 3.3 \
   --lm_beam_size 2000 \
</pre>


## Contact
In case you have any questions or inquiries, feel free to send an email to: <a href="mailto:support@moncefbenaicha.me">Moncef</a> or <a href="mailto:tugtekin.turan@iais.fraunhofer.de">Tuğtekin</a>

## Acknowledgments
This work was done during my work at Fraunhofer IAIS and it's supported by the European Union’s Horizon 2020
Research and Innovation Program under Grant Agreement No. 957017
SELMA (https://selma-project.eu).

## Paper Citation
<pre>
</pre>
