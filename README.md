# China_SOC_Prediction
This repository contains the implementation of TabTransformer_WT, a deep learning model for optimizing the estimation and projection of surface (0-20 cm) and profile (0-100 cm) soil organic carbon (SOC) in China. The model combines TabTransformer architecture with a weighted mean squared error loss function to incorporate prior knowledge of key environmental drivers.
# Overview
TabTransformer_WT is a deep learning model designed for SOC prediction and projection across China. The model:

Combines the attention mechanism of Transformer architecture with weighted loss functions
Incorporates prior knowledge of key environmental drivers
Achieves higher prediction accuracy compared to traditional machine learning and other deep learning approaches
Supports spatiotemporal modeling of SOC under different climate change scenarios
Offers enhanced interpretability through attention mechanisms and feature importance analysis

This implementation is part of the research work described in "Optimizing estimation and projection of surface and profile soil organic carbon in China" (Zhang et al., 2025).

# Setting up the environment
bash

Clone the repository

git clone https://github.com/zzbsxy/China_SOC_Prediction.git
cd TabTransformer-SOC-China

Create a conda environment

conda create -n tabtransformer-soc python=3.7
conda activate tabtransformer-soc

Install dependencies

pip install -r requirements.txt

# Key Parameters
# Model Configuration
The config/model_config.yaml file contains the following key parameters:
# TabTransformer Architecture

n_blocks: Number of Transformer blocks (default: 6)

n_heads: Number of attention heads per block (default: 8)

dim_model: Dimension of the model (default: 64)

dim_ffn: Dimension of feed-forward network (default: 128)

dropout: Dropout rate (default: 0.1)

attention_dropout: Attention dropout rate (default: 0.1)


# Weighted Loss Function

weights_0_20cm: Dictionary of feature weights for 0-20cm soil layer

tmp: Weight for temperature (default: 0.1161)

srad: Weight for shortwave radiation (default: 0.3214)

p_pet: Weight for humidity index (default: 0.3661)

npp: Weight for net primary productivity (default: 0.1964)


weights_0_100cm: Dictionary of feature weights for 0-100cm soil layer

tmp: Weight for temperature (default: 0.1034)

srad: Weight for shortwave radiation (default: 0.2672)

p_pet: Weight for humidity index (default: 0.2931)

npp: Weight for net primary productivity (default: 0.3362)



# Training Parameters

optimizer: Optimizer type (default: "adam")

initial_lr: Initial learning rate (default: 1e-4)

weight_decay: Weight decay rate (default: 1e-4)

batch_size: Batch size (default: 128)

epochs: Number of training epochs (default: 100)

lr_scheduler: Learning rate scheduler (default: "step")

lr_step_size: Step size for StepLR scheduler (default: 30)

lr_gamma: Multiplication factor for StepLR scheduler (default: 0.5)

early_stopping_patience: Patience for early stopping (default: 20)
