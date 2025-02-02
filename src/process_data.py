#!/usr/bin/env python
import os
import string
import random
import torch
import re
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset

from hyperparameters import block_size

languages = [
    "af", "am", "ar", "as", "az", "be", "bg", "bn", "bn_rom", "br", "bs", "ca",
    "cs", "cy", "da", "de", "el", "en", "eo", "es", "et", "eu", "fa", "ff", "fi",
    "fr", "fy", "ga", "gd", "gl", "gn", "gu", "ha", "he", "hi", "hi_rom", "hr",
    "ht", "hu", "hy", "id", "ig", "is", "it", "ja", "jv", "ka", "kk", "km", "kn",
    "ko", "ku", "ky", "la", "lg", "li", "ln", "lo", "lt", "lv", "mg", "mk", "ml",
    "mn", "mr", "ms", "my", "my_zaw", "ne", "nl", "no", "ns", "om", "or", "pa",
    "pl", "ps", "pt", "qu", "rm", "ro", "ru", "sa", "sc", "sd", "si", "sk", "sl",
    "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "ta_rom", "te", "te_rom",
    "th", "tl", "tn", "tr", "ug", "uk", "ur", "ur_rom", "uz", "vi", "wo", "xh",
    "yi", "yo", "zh-Hans", "zh-Hant", "zu",
]

train_n = 10000 #10000
seed = 42
buffer_size = 100000
train_file_path = "cse447-project/datasets/train.txt"
eval_file_path = "cse447-project/datasets/eval.txt"

train_data = []
for lang in tqdm(languages, desc="Processing languages"):
    counter = 0

    dataset = load_dataset('statmt/cc100', lang=lang, split='train', streaming=True, trust_remote_code=True)
    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    train_samples = shuffled_dataset.take(buffer_size)

    for item in train_samples:
        sample = item["text"]

        if len(sample) > block_size + 1:
            counter += 1
            train_data.append(sample)

        if counter >= train_n:
            break

print(len(train_data))
random.shuffle(train_data)

with open("cse447-project/datasets/train_af2.txt", "w", encoding='utf-8') as train:
    for line in train_data:
        train.write(line)