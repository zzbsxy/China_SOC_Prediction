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
git clone https://github.com/yourusername/TabTransformer-SOC-China.git
cd TabTransformer-SOC-China

Create a conda environment
conda create -n tabtransformer-soc python=3.7
conda activate tabtransformer-soc

Install dependencies
pip install -r requirements.txt
