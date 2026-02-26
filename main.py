import argparse  # бібліотка для парсингу даниї (нових)
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
    model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    adapter_path: Optional[str] = None

    # data
    train_data = "data/train.jsonl"
    val_data = "data/val.jsonl"
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
    warmup_steps = 50  # відстежання змін
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

    def __init__(self, tokenizer, max_length=1024):
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


def create_example_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    train_examples = [
        {
            "instruction": "Поясни, що таке машинне навчання.",
            "response": "Машинне навчання — це підгалузь штучного інтелекту, де алгоритми навчаються на даних без явного програмування кожного правила.",
        },
        {
            "instruction": "Як працює нейронна мережа?",
            "response": "Нейронна мережа складається з шарів нейронів. Дані проходять через вхідний шар, обробляються прихованими шарами та виводяться через вихідний шар.",
        },
        {
            "instruction": "Що таке fine-tuning?",
            "response": "Fine-tuning — це процес додаткового навчання вже натренованої моделі на специфічних даних для адаптації до конкретної задачі.",
        },
    ]

    valid_examples = [
        {
            "instruction": "Що таке градієнтний спуск?",
            "response": "Градієнтний спуск — це оптимізаційний алгоритм, який мінімізує функцію втрат шляхом ітеративного оновлення параметрів моделі.",
        }
    ]

    for split, examples in [("train", train_examples), ("val", valid_examples)]:
        path = Path(output_dir) / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"Приклад даних створено в {output_dir}")


def get_memory_usage():
    try:
        mem = mx.metal.get_active_memory() / 1e9
        peak = mx.metal.get_peak_memory() / 1e9
        return f"Пам'ять: {mem:.2f} GB, Пік: {peak:.2f} GB"
    except Exception as e:
        logger.warning(f"Не вдалося отримати інформацію про пам'ять: {e}")
        return "Пам'ять: N/A"


def save_config(config: FinetuneConfig, output_dir):
    confiq_path = Path(output_dir) / "config.json"
    with open(confiq_path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, ensure_ascii=False, indent=2)
    logger.info(f"Конфігурацію зебережено  в {confiq_path}")


def run_finetuning(config: FinetuneConfig):
    logger.info("Початок процесу fine-tuning")
    logger.info(f"Конфігурація: {config}")

    # 1. Завантаження моделі та токенізатора
    logger.info("[1/5] Завантаження моделі та токенізатора")
    model, tokenizer = load_model_utils(config.model)

    # 2. Застосування LoRa адаптерів
    logger.info("[2/5] Застосування LoRa адаптерів")
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=config.lora_layers,
        config={
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "scale": config.lora_alpha / config.lora_rank,
        }
    )

    trainable = sum(v.size for _, v in model.trainable_parameters()) / 1e6
    total = sum(v.size for _, v in model.parameters()) / 1e6
    logger.info(f"Параметри для навчання: {trainable:.2f}M / {total:.2f}M")

    # 3. Підготовка даних
    logger.info("[3/5] Підготовка даних")
    loader = DatasetLoader(tokenizer, max_length=config.max_seq_length)
    train_samples = loader.tokenize(loader.load(config.train_data))
    val_samples = None
    if config.val_data and Path(config.val_data).exists():
        val_samples = loader.tokenize(loader.load(config.val_data))

    # 4. Налаштування та запуск навчання
    logger.info("[4/5] Налаштування та запуск навчання")
    os.makedirs(config.output_dir, exist_ok=True)
    save_config(config, config.output_dir)

    training_args = TrainingArgs(
        batch_size=config.batch_size,
        iters=config.epochs * (len(train_samples) // config.batch_size),
        val_batches=len(train_samples) // config.batch_size if val_samples else 0,
        steps_per_report=config.log_every,
        steps_per_save=config.save_every,
        adapter_file=config.output_dir,
        max_seq_length=config.max_seq_length,
        grad_checkpoint=config.gradient_checkpoint
    )

    optimizer = optim.AdamW(
        learning_rate=build_schedule(
            {
                "name": "AdamW",
                "warmup": config.warmup_steps,
                "arguments": [
                    config.learning_rate, training_args.iters
                ]
            }
        )
    )

    start = time.time()
    train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=train_samples,
        val_dataset=val_samples,
    )
    elapsed = time.time() - start
    logger.info(f"Навчання завершено за {elapsed:.2f} секунд")

    # 5. Оцінка та генерація
    logger.info("[5/5] Оцінка та генерація")
    prompt = "### Instruction:\nЩо таке штучний інтелект?\n\n### Response:\n"
    response = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        verbose=False
    )
    logger.info(f"Запит: {prompt.strip()}")
    logger.info(f"Відповідь: {response.strip()}")


if __name__ == "__main__":
    run_finetuning(FinetuneConfig())
