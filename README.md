# Diffusion Models for Image Generation

## Overview

This repository contains the implementation and experiments for generating images using **Diffusion Models**, a class of generative models. The project explores three variants:
- **Unconditional Diffusion**
- **Conditional Diffusion**
- **Stable Diffusion**

The code focuses on the noising and denoising processes and showcases their effectiveness across multiple datasets.

## Features

- **Unconditional Generation**: Trained models generate images without supervision.
- **Conditional Generation**: Models generate images based on specific conditions like textual prompts.
- **Stable Diffusion**: Implements latent space compression using an autoencoder for faster and efficient training.

## Datasets

The models are trained and evaluated on:
1. **MNIST**
2. **Fashion MNIST**
3. **CIFAR10**

## Methodology

### Noising Process
Gradual addition of Gaussian noise over time, using the equation:
\[
x(t+\Delta t) = x(t) + \sigma(t) \sqrt{\Delta t} r
\]
Where:
- \( \Delta t \): Time step
- \( x(t) \): Image at time \( t \)
- \( r \sim N(0, 1) \): Standard normal random variable

### Denoising Process
The denoising equation predicts the mean to recover the original image:
\[
x(t+\Delta t) = x(t) + \sigma(T-t)^2 \frac{d}{dx} [\log p(x, T - t)] \Delta t + \sigma(T-t) \sqrt{\Delta t} r
\]
Neural networks are trained to minimize the denoising objective.

### Architectures
1. **U-Net with Gaussian Fourier Projection** for temporal embeddings.
2. Enhanced U-Net complexity for Fashion MNIST and CIFAR10.
3. **Autoencoder** for latent space compression in Stable Diffusion.

## Results

### Unconditional Diffusion
- Progressive improvement in generated images.
- Loss curves and generated images visualized across epochs.

### Conditional Diffusion
- Attention layers added to U-Net for better feature capture.
- Trained with **AdamW optimizer** and a custom learning rate scheduler.

### Stable Diffusion
- Compressed latent space leads to faster training but slightly reduced image quality.

