# Phinetuning

Advanced finetuning toolkit for Phi-2 and other language models, featuring distributed training, hyperparameter optimization, and mixed precision training.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Generation](#generation)
- [Advanced Configuration](#advanced-configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- Distributed training support
- Mixed precision training (FP16/BF16)
- 4-bit and 8-bit quantization
- Hyperparameter optimization using Optuna
- Data augmentation techniques
- Advanced text generation with various decoding strategies
- Streaming generation support
- Comprehensive logging and error handling
- Support for multiple model architectures

## Installation

```bash
git clone https://github.com/murapa96/Phinetuning.git
cd Phinetuning
pip install -r requirements.txt
```
## Usage

### Training

Basic training command:

```bash
python train.py \
  --model_path microsoft/phi-2 \
  --dataset_path your_dataset.json \
  --output_dir ./results \
  --batch_size 1 \
  --grad_accum_steps 4
```

Enable advanced features:

```bash
python train.py \
  --model_path microsoft/phi-2 \
  --dataset_path your_dataset.json \
  --output_dir ./results \
  --mixed_precision bf16 \
  --load_in_4bit \
  --distributed \
  --tune_hyperparams \
  --n_trials 10 \
  --augment_data
```

### Generation

Basic text generation:

```bash
python generate.py \
  --model_path ./results/final_model \
  --input_text "Your prompt here" \
  --max_length 200
```

Advanced generation with sampling:

```bash
python generate.py \
  --model_path ./results/final_model \
  --input_file inputs.json \
  --output_file outputs.json \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --do_sample \
  --streaming \
  --mixed_precision
```

## Advanced Configuration

### Training Arguments

- `--mixed_precision`: Choose between 'no', 'fp16', or 'bf16'
- `--load_in_4bit`: Enable 4-bit quantization
- `--load_in_8bit`: Enable 8-bit quantization
- `--distributed`: Enable distributed training
- `--tune_hyperparams`: Enable Optuna hyperparameter tuning
- `--augment_data`: Enable data augmentation
- `--max_seq_length`: Maximum sequence length (default: 2048)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--batch_size`: Per device batch size
- `--grad_accum_steps`: Gradient accumulation steps

### Generation Arguments

- `--temperature`: Controls randomness (0.0-1.0)
- `--top_k`: Top-k sampling parameter
- `--top_p`: Nucleus sampling parameter
- `--num_beams`: Number of beams for beam search
- `--repetition_penalty`: Penalize repeated tokens
- `--streaming`: Enable token-by-token generation
- `--mixed_precision`: Enable mixed precision inference
- `--load_in_4bit`: Enable 4-bit quantization for memory efficiency

## Examples

1. **Distributed training with mixed precision:**

```bash
python train.py \
  --model_path microsoft/phi-2 \
  --dataset_path your_dataset.json \
  --distributed \
  --mixed_precision bf16 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --learning_rate 2e-4 \
  --max_steps 10000
```

2. **Hyperparameter optimization:**

```bash
python train.py \
  --model_path microsoft/phi-2 \
  --dataset_path your_dataset.json \
  --tune_hyperparams \
  --n_trials 20 \
  --load_in_4bit
```

3. **Batch generation with advanced sampling:**

```bash
python generate.py \
  --model_path ./results/final_model \
  --input_file inputs.json \
  --output_file outputs.json \
  --batch_size 4 \
  --temperature 0.7 \
  --top_p 0.95 \
  --do_sample \
  --mixed_precision
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
