# MasterThesis

# Rotationally Invariant Neural Networks

## Overview
This project explores **rotationally invariant convolutional neural networks (CNNs)**, addressing the lack of established architectures that maintain invariance to rotational transformations in image processing. The work is based on recent advancements in deep learning and aims to analyze the efficiency of novel architectures proposed in scientific literature.

## Objectives
- Develop an enriched dataset by incorporating various rotated versions of images (e.g., traffic signs, letters).
- Implement new rotationally invariant architectures using **PyTorch** 
- Compare the performance of these architectures against traditional CNNs on the enhanced dataset.

## Technologies Used
- **Programming Language:** Python
- **Machine Learning Frameworks:**
  - **PyTorch** – Flexible framework for prototyping and training deep learning models.
  - **Rotationally Invariant CNN Architectures** – Based on research such as *General E(2)-Equivariant Steerable CNNs* ([reference](https://arxiv.org/pdf/2007.10588.pdf)).
- **GPU Acceleration:** Utilizing **NVIDIA GPUs** to accelerate deep learning computations through CUDA.
- **Containerization:** **Docker** for environment isolation, ensuring reproducibility and ease of collaboration.
