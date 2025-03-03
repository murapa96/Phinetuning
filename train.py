import os
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import random

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
from datasets import load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import optuna
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed,
)
from tqdm import tqdm

from trl import SFTTrainer
from huggingface_hub import interpreter_login, login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train script with enhanced capabilities')
    
    # Model and data arguments
    parser.add_argument('--model_path', type=str, default="microsoft/phi-2", help='Path to the model')
    parser.add_argument('--tokenizer_path', type=str, default="microsoft/phi-2", help='Path to the tokenizer')
    parser.add_argument('--dataset_path', type=str, default="your_dataset.json", help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default="./results", help='Directory to save results')
    
    # Training configuration
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=1, help='Per device batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum number of training steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio')
    parser.add_argument('--save_steps', type=int, default=500, help='Steps between model saves')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    
    # Mixed precision and quantization
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['no', 'fp16', 'bf16'], 
                        help='Mixed precision training mode')
    parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit quantization')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8-bit quantization')
    
    # LoRA configuration
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA r parameter')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    
    # Hyperparameter tuning
    parser.add_argument('--tune_hyperparams', action='store_true', help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of Optuna trials')
    
    # Data augmentation
    parser.add_argument('--augment_data', action='store_true', help='Enable data augmentation')
    parser.add_argument('--augment_prob', type=float, default=0.1, help='Data augmentation probability')
    
    return parser.parse_args()

def setup_distributed_training(args):
    """Initialize distributed training if enabled."""
    if args.distributed:
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl')
            logger.info(f"Initialized distributed training on rank {args.local_rank}")
            return True
    return False

def get_target_modules_for_model(model_path):
    """Determine appropriate target modules based on model architecture."""
    if "phi" in model_path.lower():
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    else:
        # Default target modules, should be adjusted based on model architecture
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

def apply_data_augmentation(example, tokenizer, augment_prob=0.1):
    """Apply simple data augmentation techniques to the text."""
    if random.random() > augment_prob:
        return example
    
    text = example["text"]
    augmentation_type = random.choice(["synonym", "deletion", "insertion", "swap"])
    
    if augmentation_type == "deletion" and len(text) > 10:
        # Random character deletion
        pos = random.randint(0, len(text) - 1)
        text = text[:pos] + text[pos+1:]
    elif augmentation_type == "swap" and len(text) > 10:
        # Swap two adjacent words
        words = text.split()
        if len(words) > 1:
            i = random.randint(0, len(words) - 2)
            words[i], words[i+1] = words[i+1], words[i]
            text = " ".join(words)
    
    example["text"] = text
    return example

def prepare_dataset(args, tokenizer):
    """Load and prepare the dataset with optional augmentation."""
    try:
        if args.dataset_path.endswith(".json"):
            dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        elif os.path.isdir(args.dataset_path):
            dataset = load_from_disk(args.dataset_path)
        else:
            raise ValueError(f"Unsupported dataset path: {args.dataset_path}")
        
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        
        if args.augment_data:
            logger.info("Applying data augmentation...")
            dataset = dataset.map(
                lambda example: apply_data_augmentation(example, tokenizer, args.augment_prob),
                desc="Augmenting data"
            )
            logger.info(f"Dataset after augmentation: {len(dataset)} samples")
        
        return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

def objective(trial, args, dataset, tokenizer):
    """Objective function for Optuna hyperparameter tuning."""
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32])
    
    # Create quantization config
    bnb_config = create_quantization_config(args)
    
    # Setup the model
    model = load_model(args, bnb_config)
    
    # Create LoRA config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=get_target_modules_for_model(args.model_path),
    )
    
    # Training arguments for this trial
    training_args = TrainingArguments(
        output_dir=f"./optuna_results/trial_{trial.number}",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=lr,
        max_steps=min(1000, args.max_steps),  # Shorter training for trials
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_ratio=args.warmup_ratio,
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    # Train and evaluate
    try:
        trainer.train()
        return trainer.state.log_history[-1].get("train_loss", float("inf"))
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        return float("inf")

def create_quantization_config(args):
    """Create the quantization configuration based on arguments."""
    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    elif args.load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None

def load_model(args, bnb_config=None):
    """Load the model with proper configuration."""
    try:
        logger.info(f"Loading model from {args.model_path}")
        device_map = {"": args.local_rank} if args.local_rank != -1 else "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            quantization_config=bnb_config, 
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # Setup for kbit training if using quantization
        if args.load_in_4bit or args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        
        # Important for training stability with phi-2
        model.config.pretraining_tp = 1 
        model.config.use_cache = False
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def main():
    """Main training function with improved error handling and features."""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    try:
        # Setup distributed training if enabled
        is_distributed = setup_distributed_training(args)
        
        # Login to Hugging Face
        try:
            interpreter_login()
        except:
            logger.warning("interpreter_login failed, trying regular login")
            login()
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare dataset
        dataset = prepare_dataset(args, tokenizer)
        
        if args.tune_hyperparams:
            logger.info("Starting hyperparameter tuning with Optuna")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, args, dataset, tokenizer), n_trials=args.n_trials)
            
            # Extract best parameters
            best_params = study.best_params
            logger.info(f"Best hyperparameters: {best_params}")
            
            # Update args with best parameters
            args.learning_rate = best_params["learning_rate"]
            args.batch_size = best_params["batch_size"]
            args.lora_r = best_params["lora_r"]
            args.lora_alpha = best_params["lora_alpha"]
        
        # Create quantization config
        bnb_config = create_quantization_config(args)
        
        # Load the model
        model = load_model(args, bnb_config)
        
        # Create LoRA configuration
        logger.info("Setting up LoRA configuration")
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=get_target_modules_for_model(args.model_path),
        )
        
        # Create training arguments
        logger.info("Setting up training arguments")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            optim="paged_adamw_32bit",
            save_steps=args.save_steps,
            logging_steps=10,
            learning_rate=args.learning_rate,
            fp16=(args.mixed_precision == "fp16"),
            bf16=(args.mixed_precision == "bf16"),
            max_grad_norm=0.3,
            max_steps=args.max_steps,
            warmup_ratio=args.warmup_ratio,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            gradient_checkpointing=True,
            remove_unused_columns=False,
        )
        
        # Create SFT Trainer
        logger.info("Initializing trainer")
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            packing=False,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        # Start training
        logger.info("Starting training")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving model to {os.path.join(args.output_dir, 'final_model')}")
        trainer.save_model(os.path.join(args.output_dir, "final_model"))
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()