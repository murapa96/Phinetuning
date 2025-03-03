# Phinetuning

A test repository to finetune phi2 models.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository is dedicated to the finetuning of phi2 models using Python. The primary goal is to test and improve the performance of these models through various experiments and adjustments.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/murapa96/Phinetuning.git
cd Phinetuning
pip install -r requirements.txt
```

## Usage

To run the finetuning process, use the following command:

```bash
python finetune.py --config config.yaml
```

Make sure to update the `config.yaml` file with your specific parameters and settings.

## Examples

Here are some examples of how to use the scripts in this repository:

1. **Finetuning a model:**
    ```bash
    python finetune.py --config configs/finetune_example.yaml
    ```

2. **Evaluating a model:**
    ```bash
    python evaluate.py --model_path models/finetuned_model.pth --data_path data/test_data.csv
    ```

## Contributing

We welcome contributions to this project. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b my-feature-branch`.
3. Make your changes and commit them: `git commit -am 'Add new feature'`.
4. Push the branch: `git push origin my-feature-branch`.
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
