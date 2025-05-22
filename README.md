# Kidney Cancer Detection

A secure, end-to-end pipeline for automated classification of kidney conditions (tumors, cysts, stones, benign lesions) from CT scans, with privacy-preserving and adversarial defenses built in.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Makefile Commands](#makefile-commands)  
7. [Sample Data](#sample-data)  
8. [Research Papers](#research-papers)
9. [Slideshow](#slideshow)  


---

## Project Overview
This repository implements a Convolutional Neural Network (ResNet-50) pipeline in PyTorch for classifying four kidney conditions using CT imagery. It includes modules for data preprocessing, mixed-precision training, evaluation, and security analysis against adversarial, poisoning, and inversion attacks.

## Features
- **High Accuracy**: 95% test accuracy on four classes  
- **Security Focus**: Tests for adversarial robustness, data poisoning, and model inversion  
- **Modular Design**: Separate managers for preprocessing, training, evaluation, and orchestration  
- **Mixed-Precision**: FP16 training support on CUDA for speed & memory efficiency  
- **CLI & Tests**: `PipelineManager.cli()` and pytest fixtures for reproducible runs

## Dataset
- **Source**: Public Kaggle repository ([Kidney CT Scan](https://www.kaggle.com/datasets/anima890/kidney-ct-scan)) plus partner hospital data  
- **Composition**: 10,000 de-identified CT images (≈2,500 per class)  
- **Preprocessing**: Windowed to kidney-specific HU range, resized to 224×224, normalized to ImageNet stats

## Installation
```zsh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

## Usage
```zsh
@python src/pipeline_manager.py \
    --data-dir path/to/kidney_ct_data \
    --batch-size 32 \
    --epochs 3 \
    --test-size 0.2 \
    --output-dir evaluation_results
```

## Makefile Commands
- `make train`  Run training and unit tests  
- `make evaluate` Run evaluation suite and generate artifacts  
- `make pipeline` Execute pytest on `tests/test_pipeline.py`  
- `make clean`  Remove build artifacts and caches  
- `make usage`  Run the full pipeline with default parameters

## Sample Data
Below are example CT images illustrating each kidney condition class in our dataset:

| Tumor | Cyst | Stone | Benign Lesion |
|:-----:|:----:|:-----:|:-------------:|
| ![Tumor Example](kidney_ct_data/Tumor/Tumor-%20(1).jpg) | ![Cyst Example](kidney_ct_data/Normal/Normal-%20(1).jpg) | ![Stone Example](kidney_ct_data/Stone/Stone-%20(1).jpg) | ![Benign Example](kidney_ct_data/Cyst/Cyst-%20(1).jpg) |

## Research Papers
- [Functional Research Paper](docs/functional_research_paper.pdf)
- [Security Research Paper](docs/security_research_paper.pdf)

## Slideshow
- [Project Slide Deck](slides/KidneyCancerDetection_Slides.pptx)