# Rotated-datasets-evaluation-pipeline

Pipeline for generating rotated image dataset variants and evaluating CNN/CyCNN models across rotation-based train-test scenarios.

This repository contains research code and experiment utilities developed for working with rotated image datasets, model evaluation logs, confusion matrices, and result analysis. It was created as part of a master's thesis project and later adapted as a public research artifact for scientific publication.

## Overview

The project focuses on evaluating how image classification models behave under rotation-based data transformations.

The repository supports the following workflow:

1. Prepare image datasets.
2. Generate fixed-angle and angle-range rotated variants.
3. Build train/test scenario configurations.
4. Ingest and validate experiment logs.
5. Analyze confusion matrices and evaluation results.
6. Generate heatmaps, learning curves, rankings, and model summaries.

The code is intended for experiments involving standard CNN models and rotation-aware or rotation-related architectures such as CyCNN variants.

## What this repository contains

- Dataset preparation utilities for MNIST, GTSRB, GTSRB_RGB, and LEGO-like datasets.
- Rotation pipeline for fixed-angle and angle-range dataset variants.
- Train/test scenario generation for rotated and merged datasets.
- Experiment log ingestion and validation.
- SQLite-based storage of experiment metadata and results.
- Confusion matrix analysis.
- Accuracy heatmap generation.
- Learning curve generation.
- Optuna experiment analysis.
- Best-model ranking and summary generation.

## Repository structure

```text
.
├── configs/
│   └── scenarios/              # Train/test scenario JSON files
├── dataset/                    # Dataset files and generated dataset variants
├── results/                    # Logs, databases, plots, heatmaps, and summaries
├── src/
│   ├── analysis/               # Result analysis, log ingestion, matrices, plots
│   ├── datasets/               # Dataset preparation utilities
│   ├── pipelines/              # Dataset rotation and scenario-generation pipeline
│   ├── scripts/                # Additional plotting/export scripts
│   └── utils/                  # Shared helpers
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Supported datasets

The repository was designed around experiments on:

- MNIST
- GTSRB
- GTSRB_RGB
- LEGO-like image datasets

Some datasets or derived files may be included directly in the repository, while others may need to be downloaded or prepared separately depending on the experiment.

## Dataset variants

The rotation pipeline can generate multiple types of dataset variants, including:

- non-rotated baseline datasets
- fixed-angle rotations, for example 30°, 45°, 60°, etc.
- angle-range rotations, for example 0–30°, 30–60°, etc.
- merged datasets combining multiple fixed-angle or angle-range variants
- train/test scenario JSON files used for systematic evaluation

Example generated variants may include names such as:

```text
rotated-30
rotated-45
rotated-0-30
merged_fixed_30
merged_fixed_45
merged_range_full_0_360
merged_range_full_0_360_plus_non_rotated
```

## Installation

Python 3.9 or newer is recommended.

Clone the repository:

```bash
git clone https://github.com/Fluorky/rotated-datasets-evaluation-pipeline.git
cd rotated-datasets-evaluation-pipeline
```

Optionally create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you use GPU-based model training or evaluation, make sure that your PyTorch installation matches your CUDA version.

## Usage

The repository exposes a Typer-based CLI.

Show available commands:

```bash
python -m src.cli --help
```

### Analyze logs and generate heatmaps

```bash
python -m src.cli analyze \
  --dataset MNIST \
  --logs-dir results/log_files_from_slave/logs \
  --output-dir results/heatmaps
```

### Ingest experiment logs

```bash
python -m src.cli ingest \
  --wsl-path "<path-to-source-logs>" \
  --local-logs results/log_files_from_slave/logs \
  --db results/db/experiment_logs.db
```

### Check training and test logs

```bash
python -m src.cli check-logs \
  --train "<path-to-training-logs>" \
  --test "<path-to-test-logs>"
```

### Analyze confusion matrices

```bash
python -m src.cli matrix-analyzer \
  --cm-root "<path-to-confusion-matrices>" \
  --db results/db/experiment_logs.db \
  --dataset MNIST \
  --metric micro
```

Optional flags:

```bash
--metric micro
--metric macro
--theta-step 15
--alpha 0.70
--top-n 50
--per-class-angles
--by-delta
```

### Generate learning curves

```bash
python -m src.cli learning-curves \
  --dataset MNIST \
  --logs-dir results/log_files_from_slave/logs \
  --output-dir results/plots
```

### Analyze Optuna logs

```bash
python -m src.cli optuna-analyze \
  --dataset MNIST \
  --optuna-logs "<path-to-optuna-logs>" \
  --output-dir results
```

### Summarize best models

```bash
python -m src.cli best-models \
  --dataset MNIST \
  --results-dir results/heatmaps
```

## Running the rotation pipeline

The rotation pipeline is implemented in:

```text
src/pipelines/rotation_pipeline.py
```

Example usage from Python:

```bash
python - <<'PY'
from src.pipelines.rotation_pipeline import run_pipeline

run_pipeline(
    base_dir="dataset/MNIST",
    dataset_name="dataset_mnist_non_rotated",
    dataset_key="MNIST",
    merged_dir_name="merged_datasets",
    max_tests=2000,
    file_format="ubyte",
)
PY
```

Adjust `base_dir`, `dataset_name`, `dataset_key`, and `file_format` for the dataset you want to process.

## Results and outputs

Depending on the command, outputs may be written to:

```text
results/db/
results/heatmaps/
results/plots/
results/exports/
results/log_files_from_slave/
```

Common output types include:

- SQLite experiment databases
- CSV summaries
- accuracy matrices
- confusion-matrix-derived rankings
- learning curves
- Optuna plots
- heatmaps
- best-model summaries

## Reproducibility notes

This repository contains research code used for dataset generation and experimental analysis. Some paths, logs, datasets, or generated artifacts may depend on the local experimental environment used during the original work.

For reproducible use, check and adjust:

- dataset paths
- log paths
- database paths
- CUDA/PyTorch versions
- available generated dataset variants
- train/test scenario JSON files
- local paths passed through CLI arguments

The recommended workflow is to keep raw data, generated datasets, logs, and result artifacts clearly separated and document any experiment-specific paths before running analyses.

## Relationship to CyCNN

This repository is not a standalone reimplementation of CyCNN. It contains dataset preparation, rotation, evaluation, and analysis utilities used around CNN/CyCNN-style experiments.

If your experiment depends on an external CyCNN implementation, install and configure that project separately, then use this repository to prepare rotated datasets and analyze experiment outputs.

## Datasets and licensing

This repository may include dataset files, generated dataset variants, experiment outputs, and analysis artifacts used during the research workflow. These files are kept in the repository to preserve the original experimental artifact structure.

The MIT License in this repository applies only to the original source code, scripts, configuration files, and project-specific documentation authored for this project.

It does **not** relicense external datasets, third-party data, pre-trained models, or files derived from external datasets. Such materials remain subject to their original licenses, terms of use, attribution requirements, and citation requirements.

### Dataset provenance

The project uses or refers to the following datasets and derived variants:

| Dataset | Usage in this repository | Source / provenance | Licensing note |
|---|---|---|---|
| MNIST | Baseline image classification dataset and source for rotated variants | MNIST handwritten digit database by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges; derived from NIST datasets | MNIST is not relicensed by this repository. Users should follow the original MNIST terms and attribution requirements. |
| GTSRB | Traffic sign classification dataset and source for rotated variants | German Traffic Sign Recognition Benchmark (GTSRB), originally provided by the Ruhr University Bochum / INI benchmark site; some workflows may use publicly available mirrors such as Kaggle | GTSRB is not relicensed by this repository. Users should follow the terms of the exact source from which they obtain the dataset. |
| GTSRB_RGB | RGB-formatted or converted GTSRB-derived dataset variant used in experiments | Derived from GTSRB preprocessing/conversion steps used in this project | This variant inherits the licensing and usage restrictions of the underlying GTSRB data. |
| LEGO-like data | Image classification dataset used for rotation-based experiments | Project-specific or third-party LEGO-like image data used during the original experiments | LEGO-derived or LEGO-like data is not relicensed by this repository. Users should verify the original source and its terms before redistribution or reuse. |
| Rotated dataset variants | Fixed-angle, angle-range, and merged variants generated by this repository | Generated from the datasets listed above using the rotation pipeline | Generated variants inherit the licensing and usage restrictions of their source datasets. |

### Generated artifacts

This repository may also contain generated files such as:

- rotated dataset variants
- merged train/test datasets
- train/test scenario files
- experiment logs
- SQLite result databases
- confusion matrices
- plots, heatmaps, rankings, and summary tables

These artifacts are included for research transparency and reproducibility. Their reuse may still be restricted if they contain, encode, transform, or derive from external datasets.

### User responsibility

Users of this repository are responsible for ensuring that they have the appropriate rights to download, use, modify, redistribute, and publish any dataset or dataset-derived artifact.

If you use this repository in academic work, please cite the relevant original dataset papers or sources in addition to citing this repository or the associated publication.

## Citation

If this repository is used in academic work, please cite the related publication or thesis when available.

A `CITATION.cff` file may be added in the future to provide machine-readable citation metadata.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
