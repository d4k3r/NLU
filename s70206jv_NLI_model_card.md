---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/d4k3r/NLU

---

# Model Card for v25523st_m61733im_s70206jv-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a blending ensemble model utilizing four encoder-only transformers and an XGBoost meta-learner. It performs binary classification for Natural Language Inference (NLI), determining whether a given premise and hypothesis pair represents entailment or contradiction.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The architecture consists of four base models: microsoft/deberta-v3-large, google/electra-large-discriminator, microsoft/deberta-v2-xxlarge, and FacebookAI/roberta-large. Base models were fine-tuned using Fast Gradient Method (FGM) adversarial training. Due to hardware memory constraints, the 1.5 billion parameter DeBERTa-v2-xxlarge model was adapted using Low-Rank Adaptation (LoRA), while the remaining models underwent full fine-tuning. An XGBoost classifier was subsequently trained on the out-of-fold validation probabilities to generate the final predictions.

- **Developed by:** Serhii Tupikin, Ilya Maltsev, Julius Vander Arend
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** XGBoost Meta-Learner Ensemble of Encoder-Only Transformers
- **Finetuned from model [optional]:** deberta-v3-large, electra-large-discriminator, deberta-v2-xxlarge, roberta-large

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** 
* [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large)
* [Electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)
* [DeBERTa-v2-xxlarge](https://huggingface.co/microsoft/deberta-v2-xxlarge)
* [RoBERTa-large](https://huggingface.co/roberta-large)
    
- **Paper or documentation:** 
* DeBERTa: [https://arxiv.org/abs/2111.09543](https://arxiv.org/abs/2111.09543)
* Electra: [https://arxiv.org/abs/2003.10555](https://arxiv.org/abs/2003.10555)
* RoBERTa: [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
    

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The base models were trained on 24,432 premise-hypothesis pairs (train.csv). The meta-learner was trained on a separate unseen validation set (dev.csv).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


**Global Ensemble Parameters:**
* Effective Batch Size: 16 (Gradient Accumulation used for deberta-v2-xxlarge due to VRAM constraints)
* Max Sequence Length: 128
* Adversarial Training: FGM enabled (epsilon=1.0)
* Epochs per base model: 5

**Base Model Specifics:**
* **DeBERTa-v3-large:** Full Fine-tune, LR: 8.35e-06, Objective: Cross Entropy
* **RoBERTa-large:** Full Fine-tune, LR: 7.80e-06, Objective: Focal Loss
* **Electra-large:** Full Fine-tune, LR: 1.51e-05, Objective: Cross Entropy
* **DeBERTa-v2-xxlarge:** LoRA (r=8, alpha=16), LR: 2.65e-4, Objective: Cross Entropy

**Meta-Learner (XGBoost):**
* max_depth: 5
* n_estimators: 100
* learning_rate: 0.1
    

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall ensemble parameter count: ~2.5 billion parameters
      - overall training time: 2 hours
      - duration per training epoch: 30 minutes
        - DeBERTa-v2-xxlarge: ~ 20 minutes
        - DeBERTa-v3-large, roberta-large, electra-large-discriminator: ~ 4 minuets
      - model size: ~11 GB tota (all base models combined)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Validation set (dev.csv) was used to evaluate model performance.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Macro F1-score
      - Cross-Entropy / Focal Loss

### Results

The ensemble model achieved a Macro F1-score of 0.9621 on the validation set (dev.csv).

## Technical Specifications

### Hardware


      - GPU: NVIDIA GeForce RTX 5090 or equivalent.
      - VRAM: 32GB needed for training, ~20 GB is ok for inference, you may lower batch size to run with less vram.
      - RAM: Minimum 32 GB.
      - Storage: Minimum 15 GB required for model weights.

### Software


* Python 3.14
* Transformers 4.57.3
* PyTorch 2.11.0+cu128
* Optuna 4.8.0
* XGBoost 3.2.0
* PEFT 0.18.1
* scikit-learn 1.8.0
* Pandas 3.0.1
* NumPy 2.4.3

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The ensemble comprises approximately 2.5 billion parameters, resulting in high inference latency. It is unsuitable for real-time or edge deployment. Inputs exceeding 128 subwords are truncated.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Hyperparameter configurations for the base models were established through a 20-trial Bayesian optimization process utilizing the Optuna framework.

For the standard large-architecture models (deberta-v3-large, electra-large, roberta-large), the search space evaluated three distinct tuning paradigms:
1. Full Fine-Tuning: Learning rate sampled log-uniformly from [5e-6, 3e-5].
2. Partial Layer Freezing: Bottom-layer freeze ratio sampled uniformly from [0.15, 0.40] (step=0.05); learning rate sampled log-uniformly from [5e-6, 3e-5].
3. Low-Rank Adaptation (LoRA): Rank (r) ∈ {4, 8, 16}, alpha ∈ {16, 32}; learning rate sampled log-uniformly from [5e-5, 5e-4].

Global categorical hyperparameters evaluated across all trials included the application of adversarial noise (FGM: True/False) and the primary objective function (Cross-Entropy vs. Focal Loss).

Hardware Constraint Methodology: Due to the  32 GB VRAM limitation of the NVIDIA RTX 5090, both full fine-tuning and partial layer freezing were computationally unfeasible for the 1.5 billion parameter microsoft/deberta-v2-xxlarge model. Consequently, its optimization space was strictly bounded to the LoRA adaptation strategy.
