import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from tqdm.notebook import tqdm

from trl import SFTTrainer
from huggingface_hub import interpreter_login

# Definir los argumentos de la línea de comandos
parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--model_path', type=str, default="microsoft/phi-2", help='Path to the model')
parser.add_argument('--tokenizer_path', type=str, default="microsoft/phi-2", help='Path to the tokenizer')
parser.add_argument('--dataset_path', type=str, default="your_dataset.json", help='Path to the dataset')

# Parse the command line arguments
args = parser.parse_args()

interpreter_login()

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_double_quant=False,
    )
device_map = {"": 0}

#Download model
model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        quantization_config=bnb_config, 
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True
    )

model.config.pretraining_tp = 1 
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    target_modules=['lm_head.linear', 'transformer.embd.wte'], # is this correct?
    bias="none",
    task_type="CAUSAL_LM", 
)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500, #CHANGE THIS IF YOU WANT IT TO SAVE LESS OFTEN. I WOULDN'T SAVE MORE OFTEN BECAUSE OF SPACE
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    max_grad_norm=.3,
    max_steps=10000,
    warmup_ratio=.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

model.config.use_cache = False

dataset = load_dataset("json", data_files=args.dataset_path, split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()