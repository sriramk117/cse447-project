#!/usr/bin/env python
import os
import random
import torch
from tqdm import tqdm
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

train_n = 100_000
seed = 42
buffer_size = 10_000
write_batch_size = 10_000

train_file_path = "cse447-project/datasets/train.txt"
block_size = 128

with open(train_file_path, "w", encoding='utf-8') as train_file:
    buffer = []

    for lang in tqdm(languages, desc="Processing languages"):
        sample_count = 0

        dataset = load_dataset('statmt/cc100', lang=lang, split='train', streaming=True, trust_remote_code=True)

        for item in dataset:
            sample = item["text"]

            if len(sample) > block_size + 1:
                buffer.append(sample)
                sample_count += 1

                if len(buffer) >= write_batch_size:
                    train_file.writelines(buffer)
                    buffer = []

            if sample_count >= train_n:
                break

    if buffer:
        train_file.writelines(buffer)

with open(train_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(len(lines))
random.shuffle(lines)
with open(train_file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

