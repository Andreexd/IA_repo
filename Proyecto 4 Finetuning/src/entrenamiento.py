import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "../models/lora"

TRAIN_FILE = "../data/processed/train.jsonl"
VAL_FILE = "../data/processed/val.jsonl"

MAX_LENGTH = 128
EPOCHS = 4
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4

def format_prompt(example):
    text = (
        f"### Instrucción:\n{example['instruction']}\n\n"
        f"### Respuesta:\n{example['output']}</s>"
    )
    return {"text": text}

def main():
    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "val": VAL_FILE
        }
    )

    dataset = dataset.map(format_prompt)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    device = "cpu"
    dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,   
        warmup_steps=5,
        logging_steps=10,
        save_steps=500,
        fp16=False,
        dataloader_pin_memory=False,
        do_eval=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
