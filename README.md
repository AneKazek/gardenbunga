# Project GardenBunga: Efficient Hybrid Sequence Modeling for Indonesian Speech Synthesis via Multilingual Knowledge Transfer

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11"></a>
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.8.0-ee4c2c.svg" alt="PyTorch 2.8.0"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.8-76b900.svg" alt="CUDA 12.8"></a>
  <a href="https://github.com/state-spaces/mamba"><img src="https://img.shields.io/badge/Mamba-2.3.1-6f42c1.svg" alt="Mamba 2.3.1"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Language-Indonesian%20TTS-orange.svg" alt="Indonesian TTS">
  <img src="https://img.shields.io/badge/Research-Hybrid%20DiT--Mamba-black.svg" alt="Hybrid DiT-Mamba">
</p>

<p align="center">
  A research-focused F5-TTS fork for efficient Indonesian speech synthesis with conservative Hybrid DiT-Mamba integration.
</p>

## Authors

| Name | NRP |
| --- | --- |
| Muhammad Dzaky Haidar | 5054251039 |
| Benedictus Ryu Gunawan | 5054251001 |
| Muhammad Irzam Hafis Fabiansyah | 5054251024 |

## Overview

Project GardenBunga is a research fork of [F5-TTS](https://github.com/SWivid/F5-TTS) centered on Indonesian speech synthesis through multilingual knowledge transfer and efficient hybrid sequence modeling.

The main idea in this repository is deliberately conservative: instead of replacing the whole backbone, we only swap a very small number of DiT attention blocks with Mamba SSM blocks while keeping the rest of the F5-TTS training and inference pipeline as intact as possible. The current default conservative configuration uses a 4-block Hybrid DiT-Mamba setup.

This repository keeps the upstream package and CLI naming for compatibility. That means the project is called **GardenBunga**, but Python package and console commands still use the upstream `f5-tts` naming convention.

## Research Focus

- Conservative Hybrid DiT-Mamba integration for Indonesian TTS
- Multilingual knowledge transfer from pretrained F5-TTS checkpoints
- Minimal architectural disruption for faster stabilization
- Kaggle-friendly finetuning workflow with local dataset paths
- Lightweight inference smoke testing after training

## Current Default Hybrid Setup

The current conservative configuration lives in [F5TTS_v1_Base_Mamba_Conservative.yaml](./src/f5_tts/configs/F5TTS_v1_Base_Mamba_Conservative.yaml).

Key points:

- Backbone: `DiT`
- Hybrid blocks: `mamba_block_ids: [7, 11, 14, 18]`
- Mamba mode: bidirectional
- Mamba init: `alpha = 0.0` for conservative blending startup
- Vocoder path: standard F5-TTS pipeline

This means the project is already configured to use Mamba in a limited, controlled way rather than performing a full architectural replacement.

## Repository Layout

- [src/f5_tts/configs/F5TTS_v1_Base_Mamba_Conservative.yaml](./src/f5_tts/configs/F5TTS_v1_Base_Mamba_Conservative.yaml)  
  Default conservative hybrid configuration.

- [src/f5_tts/model/hybrid_mamba.py](./src/f5_tts/model/hybrid_mamba.py)  
  Hybrid Mamba mixer implementation for the conservative swap.

- [src/f5_tts/train/train.py](./src/f5_tts/train/train.py)  
  Main training entrypoint.

- [src/f5_tts/train/datasets/prepare_csv_wavs.py](./src/f5_tts/train/datasets/prepare_csv_wavs.py)  
  Dataset preparation script for custom CSV metadata.

- [src/f5_tts/infer](./src/f5_tts/infer)  
  Inference utilities, CLI, and Gradio app.

- [notebook/traineo1_kaggle_clean.ipynb](./notebook/traineo1_kaggle_clean.ipynb)  
  Kaggle-oriented notebook for clone, install, dataset preparation, finetuning, and inference smoke test.

- [deep-research-report.md](./deep-research-report.md)  
  Design rationale and conservative hybrid integration notes.

## Installation

### 1. System prerequisites

Recommended for Linux or WSL:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3.11 python3.11-venv python3.11-dev build-essential
```

### 2. Clone the repository

```bash
git clone https://github.com/AneKazek/gardenbunga.git
cd gardenbunga
```

### 3. Create a Python 3.11 environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel "setuptools<82"
```

### 4. Install PyTorch with CUDA 12.8

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128
```

### 5. Install the project

Base install:

```bash
pip install -e .
```

Install with hybrid Mamba extras:

```bash
pip install -e ".[mamba]"
```

Optional evaluation dependencies:

```bash
pip install -e ".[eval]"
```

### 6. Verify the environment

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda :", torch.version.cuda)
print("abi  :", torch.compiled_with_cxx11_abi())
print("gpu  :", torch.cuda.device_count())
PY
```

## Kaggle Workflow

For the most reproducible workflow in this project, use:

- [notebook/traineo1_kaggle_clean.ipynb](./notebook/traineo1_kaggle_clean.ipynb)

That notebook is designed to:

1. clone the repo inside Kaggle,
2. create a Python 3.11 virtual environment in `/kaggle/temp`,
3. install matching PyTorch and Mamba wheels,
4. use an already-downloaded Kaggle dataset by local path,
5. prepare the dataset,
6. finetune with the conservative hybrid config,
7. run an inference smoke test.

## Dataset Preparation

This project supports custom dataset preparation through a CSV file with the required header:

```text
audio_file|text
```

Example:

```text
/absolute/path/to/audio_0001.wav|Halo, selamat datang di Project GardenBunga.
/absolute/path/to/audio_0002.wav|Ini adalah contoh metadata untuk fine-tuning.
```

Prepare the dataset with:

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /path/to/output_dataset --workers 4
```

For training with the built-in loader, the prepared dataset is typically placed under:

```text
data/<dataset_name>_pinyin
```

## Training

The main training entrypoint is:

- [train.py](./src/f5_tts/train/train.py)

The default hybrid experiment in this repository uses:

- [F5TTS_v1_Base_Mamba_Conservative.yaml](./src/f5_tts/configs/F5TTS_v1_Base_Mamba_Conservative.yaml)

Example training command:

```bash
accelerate launch src/f5_tts/train/train.py \
  --config-name F5TTS_v1_Base_Mamba_Conservative.yaml \
  datasets.name=tts_indo \
  datasets.batch_size_per_gpu=8000 \
  datasets.max_samples=64 \
  datasets.num_workers=4 \
  optim.epochs=10 \
  optim.learning_rate=1e-5 \
  optim.grad_accumulation_steps=2 \
  ckpts.save_dir=ckpts/F5TTS_v1_Base_Mamba_Conservative_vocos_pinyin_tts_indo \
  ckpts.logger=null
```

Useful notes:

- The trainer now tolerates optimizer-state mismatch when the architecture changes, so resuming from an older 1-layer hybrid checkpoint into the current 2-layer setup is safer.
- For early-stage finetuned checkpoints, `use_ema=False` may work better during inference.
- If you want a GUI-based training flow, the upstream-compatible Gradio finetune app still exists.

## Inference

There are two practical routes in this repository:

### 1. Notebook-based inference smoke test

The Kaggle notebook includes an inference test cell that:

- resolves a checkpoint,
- ensures `vocab.txt` exists,
- picks a reference audio from metadata,
- synthesizes output text,
- saves the generated waveform.

### 2. Upstream-compatible CLI and Gradio

Even though this repository is branded as GardenBunga, the CLI names remain:

```bash
f5-tts_infer-cli --help
f5-tts_infer-gradio --help
f5-tts_finetune-gradio
```

For more inference details, see:

- [src/f5_tts/infer/README.md](./src/f5_tts/infer/README.md)

## Recommended Quick Start

If you want the shortest path to a working experiment:

1. Install Python 3.11, PyTorch 2.8, and project dependencies.
2. Use [notebook/traineo1_kaggle_clean.ipynb](./notebook/traineo1_kaggle_clean.ipynb) if you are on Kaggle.
3. Prepare your CSV dataset with absolute audio paths.
4. Train with [F5TTS_v1_Base_Mamba_Conservative.yaml](./src/f5_tts/configs/F5TTS_v1_Base_Mamba_Conservative.yaml).
5. Run the notebook inference cell or use upstream CLI utilities for smoke testing.

## Evaluation

Evaluation utilities from the upstream project are still available in:

- [src/f5_tts/eval](./src/f5_tts/eval)

This includes infrastructure for objective evaluation such as WER, similarity, and MOS-related workflows depending on your setup.

## Compatibility Notes

- Package name in `pyproject.toml` is still `f5-tts`.
- Console entrypoints still use `f5-tts_*`.
- This is intentional, to preserve compatibility with the upstream ecosystem while extending the repository for GardenBunga experiments.

## Acknowledgements

This repository builds directly on the excellent upstream work:

- [F5-TTS](https://github.com/SWivid/F5-TTS)
- [E2-TTS](https://arxiv.org/abs/2406.18009)
- [Mamba](https://github.com/state-spaces/mamba)
- [Vocos](https://huggingface.co/charactr/vocos-mel-24khz)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)

We gratefully acknowledge the original F5-TTS authors and contributors for the foundation that made this research fork possible.

## License

This repository is distributed under the MIT License. See [LICENSE](./LICENSE).
