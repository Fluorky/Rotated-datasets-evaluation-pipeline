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



# CyCNN - Local Installation Guide (No Docker)

This is a step-by-step guide for running the [CyCNN](https://github.com/mcrl/CyCNN) project **without Docker**, using a local Python environment.

---

## 🔧 Requirements

- Python 3.6 or 3.7 *(recommended)*
- pip & virtualenv *(optional but recommended)*
- Git
- CPU or GPU (CUDA 10.x for GPU acceleration)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mcrl/CyCNN.git
cd CyCNN
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Create a `requirements.txt` file:

```txt
torch==1.0.1
torchvision==0.2.2
scikit-learn
pandas
matplotlib
scipy
```

Then install:

```bash
pip install -r requirements.txt
```

---

## Prepare Dataset

CyCNN uses the **MNIST-C** dataset. Clone it and move it to the `data/` directory:

```bash
git clone https://github.com/google-research/mnist-c.git
mv mnist-c data/
```

---

## Run the Model

### Train a Model

```bash
python main.py --train --model=cyvgg19 --dataset=mnist --polar-transform=linearpolar
```

Example Output:
```
configuration:  {'model': 'cyvgg19', 'train': True, 'test': False, 'polar_transform': 'linearpolar', 'augmentation': None, 'data_dir': './data', 'batch_size': 128, 'num_epochs': 9999999, 'lr': 0.05, 'dataset': 'mnist', 'redirect': False, 'early_stop_epochs': 15, 'test_while_training': False}
Using device:  cuda
1 devices available
# Parameters: 20559.2K
54000 Train data. 6000 Validation data. 10000 Test data.
===> Training mnist-cyvgg19-linearpolar begin
[Epoch 0] Train Loss: 0.508112
[Epoch 0] Validation loss: 0.1685, Accuracy: 5669/6000 (94.48%)
Saving model checkpoint to saves/mnist-cyvgg19-linearpolar.pt
...
[Epoch 52] Train Loss: 0.000117
[Epoch 52] Validation loss: 0.0367, Accuracy: 5966/6000 (99.43%)
Training Done!
```

### Evaluate a Model

```bash
python main.py --test --model=cyvgg19 --dataset=mnist --polar-transform=linearpolar
```

Example Output:
```
configuration:  {'model': 'cyvgg19', 'train': False, 'test': True, 'polar_transform': 'linearpolar', 'augmentation': None, 'data_dir': './data', 'batch_size': 128, 'num_epochs': 9999999, 'lr': 0.05, 'dataset': 'mnist', 'redirect': False, 'early_stop_epochs': 15, 'test_while_training': False}
Using device:  cuda
1 devices available
# Parameters: 20559.2K
54000 Train data. 6000 Validation data. 10000 Test data.
===> Testing mnist-cyvgg19-linearpolar with rotated dataset begin
Test loss: 1.0565, Accuracy: 8343/10000 (83.43%)
```

---

## Outputs

- Trained models: `saves/`
- Performance plots: `plots/`

---

## Notes

- Code may require minor fixes for newer PyTorch versions (e.g., deprecation of `Variable`).
- Ensure that data tensors and weights are of the same dtype.

---