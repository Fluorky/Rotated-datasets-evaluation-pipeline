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

## Project Structure

- `src/` – model and training code
- `src/cycnn.py` – main training and evaluation script
- `data/` – dataset folder
- `models/` – saved models
- `plots/` – accuracy/loss plots

---

## Prepare Dataset

CyCNN uses the **MNIST-C** dataset. Clone it and move it to the `data/` directory:

```bash
git clone https://github.com/google-research/mnist-c.git
mv mnist-c data/
```

---

## Run the Model

### Train

```bash
python src/cycnn.py --dataset_path ./data/mnist-c --model_type cyc --epochs 30
```

Available model types:

- `fc` – Fully connected
- `cnn` – Standard CNN
- `cyc` – CyCNN (default)

Optional parameters:

- `--batch_size` (default: 128)
- `--device` (`cpu` or `cuda`)

### Evaluate

To run evaluation only (assuming the model is trained):

```bash
python src/cycnn.py --dataset_path ./data/mnist-c --model_type cyc --eval
```

---

## Outputs

- Trained models: `models/`
- Performance plots: `plots/`

---

## 🛠 Notes

- Code may require minor fixes for newer PyTorch versions (e.g., deprecation of `Variable`).
- Ensure that data tensors and weights are of the same dtype.

---