import argparse # бібліотка для парсингу даниї (нових)
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, generate
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import TrainingArgs, train, evaluate
from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers
from mlx_lm.utils import load as load_model_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger(__name__)

@dataclass
class FinetuneConfig:
    # model
    model= "mlx-community/Llama-3.2-3B-Instruct-4bit"
    adapter_path: Optional[str] = None

    # data
    train_data = "data/train.json"
    val_data = "data/val.json"
    max_seq_length = 1024

    # LoRa params
    lora_rank = 8
    lora_alpha = 16.0
    lora_dropout = 0.05
    lora_layers = 16

    # Train params
    epochs = 3
    batch_size = 2
    learning_rate = 2e-4
    warmup_steps = 50 # відстежання змін
    gradient_checkpoint = True

    # Saving params
    output_dir = "./finetuned_model"
    save_every = 100
    log_every = 10

    # Generate
    max_new_tokens = 256
    temperature = 0.7

class DatasetLoader:
    CHAT_TEMPLATE = "<|system|>{system}<|end|>\n<|user|>{user}<|end|>\n<|assistant|>{assistant}<|end|>"
    INSTRUCT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

    def __init__(self, tokenizer, max_length = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"Failed to parse line {line_num}: {e}")
        logger.info(f"Loaded {len(samples)} samples")
        return samples

    def format_sample(self, sample):
        if "messages" in sample:
            parts = []
            for message in sample["messages"]:
                role, content = message.get("role", ""), message.get("content", "")
                parts.append(f"<|{role}|>{content}<|end|>")
            return "\n".join(parts)

        elif "instructions" in sample and "response" in sample:
            return self.INSTRUCT_TEMPLATE.format(
                instruction=sample["instructions"],
                response=sample["response"]
            )
        elif "text" in sample:
            return sample["text"]
        else:
            raise ValueError(f"Невідомий формат прикладу")

    def tokenize(self, samples: list[dict]) -> list[dict]:
        tokenized = []
        skipped = 0
        for sample in samples:
            text = self.format_sample(sample)
            tokens = self.tokenizer.encode(text)

            if len(tokens) > self.max_length:
                skipped += 1
                continue

            tokenized.append(tokens)

        if skipped:
            logger.warning(f"Пропущено {skipped} прикладів. Через перевищення ліміту опрацювання!")

        logger.info(f"Токенізовано {len(tokenized)} прикладів")


if __name__ == "__main__":
    print("Hello World")