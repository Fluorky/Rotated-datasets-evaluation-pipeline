# Rotated Datasets Evaluation Pipeline

This repository contains a research artifact for generating rotated image dataset variants and evaluating CNN/CyCNN models across rotation-based train-test scenarios.

It supports dataset preparation, fixed-angle and angle-range rotations, train/test scenario generation, log ingestion, confusion-matrix analysis, ranking generation, and result visualization.

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
- Core tests for result logic, parser behavior, dataset conversion, and pipeline safety.

## Repository structure

```text
.
├── .github/workflows/          # GitHub Actions CI
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
├── tests/                      # Core tests for result logic and pipeline safety
├── main.py                     # CLI entrypoint wrapper
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Test/development dependencies
├── CITATION.cff                # Machine-readable citation metadata
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

Python 3.12 is recommended for the current public artifact and CI configuration.

Clone the repository:

```bash
git clone https://github.com/Fluorky/Rotated-datasets-evaluation-pipeline.git
cd Rotated-datasets-evaluation-pipeline
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

For development and tests:

```bash
pip install -r requirements-dev.txt
```

If you use GPU-based model training or evaluation, make sure that your PyTorch installation matches your CUDA version.

## Usage

The repository exposes a Typer-based CLI. The recommended entrypoint is:

```bash
python main.py --help
```

You can also run the CLI module directly:

```bash
python -m src.cli --help
```

### Analyze logs and generate heatmaps

```bash
python main.py analyze \
  --dataset MNIST \
  --logs-dir results/log_files_from_slave/logs \
  --output-dir results/heatmaps
```

### Ingest experiment logs

```bash
python main.py ingest \
  --wsl-path "<path-to-source-logs>" \
  --local-logs results/log_files_from_slave/logs \
  --db results/db/experiment_logs.db
```

### Check training and test logs

```bash
python main.py check-logs \
  --train "<path-to-training-logs>" \
  --test "<path-to-test-logs>"
```

### Analyze confusion matrices

```bash
python main.py matrix-analyzer \
  --cm-root "<path-to-confusion-matrix-root>" \
  --db results/db/experiment_logs.db \
  --dataset MNIST \
  --metric micro
```

Useful optional flags:

```bash
--metric micro
--metric macro
--theta-step 15
--alpha 0.70
--top-n 50
--per-class-angles
--by-delta
--clear
```

### Generate learning curves

```bash
python main.py learning-curves \
  --dataset MNIST \
  --logs-dir results/log_files_from_slave/logs \
  --output-dir results/plots
```

### Analyze Optuna logs

```bash
python main.py optuna-analyze \
  --dataset MNIST \
  --optuna-logs "<path-to-optuna-logs>" \
  --output-dir results
```

### Summarize best models

```bash
python main.py best-models \
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

By default, the pipeline avoids overwriting complete existing outputs. Use the relevant CLI or Python-level overwrite option only when you intentionally want to regenerate outputs.

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

## Testing

The repository includes core tests for result logic and pipeline safety.

Run tests locally:

```bash
python -m pytest
```

Run tests with coverage diagnostics:

```bash
python -m pytest --cov=src --cov-report=term-missing
```

The tests focus on the parts of the repository most likely to affect scientific results:

- micro and macro accuracy from confusion matrices
- zero-support class handling in macro accuracy
- dataset and metric scoping in SQLite evaluation rows
- matrix ingestion from `confusion_matrix.npy`
- rotation interval parsing and Δθ binning
- log parsing for training and test logs
- rotation pipeline input validation and overwrite safety
- GTSRB_RGB conversion validation

## Project status

[![Tests](https://github.com/Fluorky/Rotated-datasets-evaluation-pipeline/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/Fluorky/Rotated-datasets-evaluation-pipeline/actions/workflows/ci.yml)

This repository includes automated smoke tests and core logic tests for metric computation, log parsing, dataset/metric scoping, rotation-angle parsing, database ingestion, and preprocessing safeguards.

## Reproducibility notes

This repository contains research code used for dataset generation and experimental analysis.

Some paths, logs, datasets, or generated artifacts may depend on the local experimental environment used during the original work.

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

This repository is not a standalone reimplementation of CyCNN.

It contains dataset preparation, rotation, evaluation, and analysis utilities used around CNN/CyCNN-style experiments. If your experiment depends on an external CyCNN implementation, install and configure that project separately, then use this repository to prepare rotated datasets and analyze experiment outputs.

## Datasets and licensing

This repository may include dataset files, generated dataset variants, experiment outputs, and analysis artifacts used during the research workflow.

These files are kept in the repository to preserve the original experimental artifact structure. The MIT License in this repository applies only to the original source code, scripts, configuration files, and project-specific documentation authored for this project. It does **not** relicense external datasets, third-party data, pre-trained models, or files derived from external datasets.

Such materials remain subject to their original licenses, terms of use, attribution requirements, and citation requirements.

### Dataset provenance

The project uses or refers to the following datasets and derived variants:

| Dataset | Usage in this repository | Source / provenance | Licensing note |
|---|---|---|---|
| MNIST | Baseline image classification dataset and source for rotated variants. | MNIST handwritten digit database. | Not relicensed by this repository. |
| GTSRB | Traffic sign classification dataset and source for rotated variants. | German Traffic Sign Recognition Benchmark. | Not relicensed by this repository. |
| GTSRB_RGB | RGB-formatted GTSRB-derived variant. | Derived from GTSRB preprocessing. | Inherits GTSRB restrictions. |
| LEGO-like data | Image classification dataset used for rotation experiments. | Project-specific or third-party image data. | Verify original source terms. |
| Rotated variants | Fixed-angle, angle-range, and merged variants. | Generated by this pipeline. | Inherit source dataset restrictions. |

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

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). 

If this repository is used in academic work, please cite the related publication or thesis when available, and use the metadata from `CITATION.cff` for citing this software artifact.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
