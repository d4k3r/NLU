# COMP34812 Natural Language Understanding Coursework

**Track A: Natural Language Inference (NLI)**

Given a premise and a hypothesis, determine if the hypothesis is true based on the premise.

This repository contains two solutions from different approach categories:
- **Category B (Non-Transformer):** RWKV-7 recurrent neural network
- **Category C (Transformer):** XGBoost Meta-Learner Ensemble of Encoder-Only Transformers

## Project Structure

```
NLU/
├── RWKV7_NLI_RTX5090.ipynb    # Category B: RWKV-7 (training + evaluation + demo)
├── transformer.ipnyb.ipynb     # Category C: DeBERTa ensemble (training + demo)
├── Data/
│   ├── train.csv               # Training data (24K+ pairs)
│   ├── dev.csv                 # Development data (6K+ pairs)
│   └── test.csv                # Test data
├── rwkv/                       # RWKV-7 trained model directory
│   ├── backbone/               # LoRA adapters and config
│   ├── classifier.pt           # Classification head weights
│   └── config.pt               # Model configuration
├── saved_models/               # Transformer ensemble models directory
│   ├── robera-large_best.pt    # (download from Google Drive if needed)
│   ├── google_electra-large-descriminator_best.pt
│   ├── microsoft_deberta-v3-large_best.pt
│   └── microsoft_deberta-v2-xxlarge_best.pt
└── README.md
```

## Requirements

**Core Dependencies:**
```bash
# PyTorch with CUDA support (adjust cu128 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Hugging Face libraries
pip install transformers peft accelerate

# ML and data processing
pip install scikit-learn pandas numpy matplotlib tqdm

# For Transformer ensemble (Category C)
pip install optuna xgboost joblib

# For RWKV-7 (Category B)
pip install flash-linear-attention
```

**Tested Versions:**
- Python 3.10+
- PyTorch 2.0+
- transformers >= 4.40
- peft >= 0.10
- scikit-learn >= 1.3
- xgboost >= 2.0

---

## Category B: RWKV-7 (RWKV7_NLI_RTX5090.ipynb)

### Overview
RWKV-7 "Goose" (7.2B parameters) is a recurrent architecture with linear time complexity and constant memory usage. Fine-tuned with LoRA and BF16 mixed precision.

### How to Run

**Training (Sections 1-12):**
1. Update paths in `CONFIG` (Section 2)
2. Run cells sequentially from Section 1 to Section 12
3. Best model is saved to `output_dir`

**Evaluation / Demo (Section 13 - Standalone):**
1. Ensure saved model directory exists with `backbone/`, `classifier.pt`, `config.pt`
2. Update `model_path` and `eval_data_path` in `EVAL_CONFIG`
3. Run only Section 13 - no other cells required

### Results (Dev Set)
| Metric | Score |
|--------|-------|
| F1 Score | 0.9419 |
| Accuracy | 0.9399 |
| MCC | 0.8796 |

---

## Category C: DeBERTa Ensemble (transformer.ipynb)

### Overview
Blending ensemble model utilizing four encoder-only transformers and an XGBoost meta-learner

**Base Models:**
- DeBERTa-v3-large (Full Fine-tune, Cross Entropy Loss)
- DeBERTa-v2-xxlarge (LoRA fine-tuning, Cross Entropy Loss)
- Electra-large (Full Fine-tune, Cross Entropy Loss)
- RoBERTa-large (Full Fine-tune, Focal Loss)

**Key Techniques:**
- Blending Ensemble Architecture (NLIBlendingEnsemble): Orchestrates feature extraction and the training pipeline across four base transformers (DeBERTa-v3, Electra, DeBERTa-v2-XXL, RoBERTa).

- XGBoost Meta-Learner: Utilizes a gradient-boosted tree framework to optimize weighting of base model predictions.

- Low-Rank Adaptation (LoRA): Parameter-Efficient Fine-Tuning (PEFT) applied to the 1.5-billion parameter DeBERTa-v2-xxlarge model to allow fine tuning with our hardware constraints.

- Fast Gradient Method (FGM): Injects adversarial noise into the embedding layer during training to mitigate overfitting and improve generalization on unseen data.

- Bayesian Hyperparameter Optimization (Optuna): Evaluates a 20-trial search space to determine learning rates, loss type, FGM suitability, and training strategy (LoRa, Full Fine-tune, or BOTx layer freezing)
    - Within LoRa strategy, Hyperparameter search optimizes the LoRa rank and alpha. 
    - Within BOTx layer freezing strategy, Hyperparameter search optimizes x (bottom x percent of model layers frozen).

- Gradient Accumulation: Accumulates gradients across micro-batches to achieve an effective batch size of 16 for large-parameter models without exceeding memory capacity.

### How to Run

**Training:**
1. Run cells 0-5 to define imports and classes
2. Cell 6 contains Optuna hyperparameter search (optional)
3. Cell 9 defines final model configurations
4. Run ensemble training with `NLIBlendingEnsemble`
5. Models saved to `./saved_models/`

**Demo / Inference:**
1. Instantiate `NLIBlendingEnsemble` with model configs
2. Call `ensemble.predict(test_csv_path='path/to/test.csv', output_filename='predictions.csv')`

### Results (Dev Set)
| Metric | Score  |
|--------|--------|
| F1 Score | 0.9621 |

---

## Trained Models

Models exceed 10MB and are stored on OneDrive:

| Model | Category | Link |
|-------|----------|------|
| RWKV-7 NLI | B (Non-Transformer) | [Download](https://drive.google.com/file/d/1Wpu_4Xfw6UwQj02oZfzR_b-G4tNQQ8rz/view?usp=sharing) |
| DeBERTa Ensemble | C (Transformer) | [Download](https://drive.google.com/file/d/1rvpgO1mVpcp-9HiECRXjeFFXPgjGG1sf/view?usp=sharing) |

---

## Attribution

### Data
Training and evaluation data provided by **COMP34812 Natural Language Understanding** coursework, University of Manchester.

### Papers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| RWKV-7 "Goose" with Expressive Dynamic State Evolution | Peng, B. et al. | 2025 | [arXiv:2503.14456](https://arxiv.org/abs/2503.14456) |
| LoRA: Low-Rank Adaptation of Large Language Models | Hu, E. et al. | 2021 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| Focal Loss for Dense Object Detection | Lin, T.Y. et al. | 2017 | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) |

### Code and Libraries

| Library | Purpose | Repository | License |
|---------|---------|------------|---------|
| Hugging Face Transformers | Pre-trained models, tokenizers | https://github.com/huggingface/transformers | Apache 2.0 |
| Hugging Face PEFT | LoRA fine-tuning | https://github.com/huggingface/peft | Apache 2.0 |
| Flash Linear Attention | RWKV-7 implementation | https://github.com/fla-org/flash-linear-attention | MIT |
| Optuna | Hyperparameter optimization | https://github.com/optuna/optuna | MIT |

### Pre-trained Model Weights
- RWKV-7 7.2B Goose: https://huggingface.co/fla-hub/rwkv7-7.2B-g0a
- DeBERTa-v3-large: https://huggingface.co/microsoft/deberta-v3-large
- DeBERTa-v2-xxlarge: https://huggingface.co/microsoft/deberta-v2-xxlarge
- Electra-Large-Descriminator: https://huggingface.co/google/electra-large-discriminator
- RoBERTa-large: https://huggingface.co/roberta-large
