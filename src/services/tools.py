# File: Tools.py, Project: CrossLingualNerFromText
# Created by Moncef Benaicha
# Contact: support@moncefbenaicha.me

import numpy as np
import os
import random
import torch


def data_portion_selection(data: list, k: int):
    k = int(round(k / 100 * len(data)))
    return random.sample(data, k)


def init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_device(use_gpu=True):
    device = "cpu"
    device_name = None
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    return device, device_name


def correct_audio_path(dataset, audio_path):
    for utterance in dataset:
        utterance[3] = os.path.join(audio_path, utterance[3])