from typing import Optional

import typer
import os
from src.analysis.learning_matrix import process_dataset
# future: from src.db.handler import show_table
# future: from src.pipelines.rotation_pipeline import run_pipeline
# future: from src.utils.log_ingestor import ingest_logs
from src.analysis.log_ingestor import ingest_logs
from src.analysis.log_checker import check_test_logs, check_training_logs
from src.analysis import matrix_analyzer as ma
from src.analysis.learning_curves import generate_learning_curves
from src.analysis.best_models import summarize_best_models
from src.pipelines.rotation_pipeline import run_pipeline
from src.analysis.optuna_analyzer import (
    generate_optuna_learning_curves,
    generate_optuna_heatmaps,
)
# import src.datasets.gtsrb as gtsrb
# import src.datasets.gtsrb_rgb as gtsrb_rgb
# import src.datasets.lego as lego
from contextlib import contextmanager

from pathlib import Path

app = typer.Typer(help="cydata-tools CLI")

@app.command("analyze")
def analyze_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name, e.g., MNIST"),
    logs_dir: Path = typer.Option("results/log_files_from_slave/logs", help="Base logs directory"),
    output_dir: Path = typer.Option("results/heatmaps", help="Output directory for heatmaps")
):
    """Analyze logs and generate accuracy heatmaps."""
    process_dataset(dataset_name=dataset, logs_base=logs_dir, output_base=output_dir)



@app.command("ingest")
def ingest_cmd(
    wsl_path: str = typer.Option(..., "--wsl-path", help="Path to logs on WSL"),
    local_logs: Path = typer.Option("results/log_files_from_slave/logs", help="Local logs path"),
    db_path: Path = typer.Option("results/db/experiment_logs.db", help="SQLite DB path"),
    overwrite_logs: bool = typer.Option(False, "--overwrite-logs", help="Force re-sync from WSL"),
    overwrite_db: bool = typer.Option(False, "--overwrite-db", help="Overwrite DB entries")
):
    """Sync logs from WSL and ingest into database."""
    ingest_logs(
        wsl_source=wsl_path,
        local_logs_path=str(local_logs),
        db_path=str(db_path),
        overwrite_logs=overwrite_logs,
        overwrite_db=overwrite_db
    )

@app.command("check-logs")
def check_logs_cmd(
    train_path: Path = typer.Option(
        Path(r"\\wsl$\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs\json_MNIST\train"),
        "--train",
        help="Path to training logs directory"
    ),
    test_path: Path = typer.Option(
        Path(r"\\wsl$\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs\json_MNIST\test"),
        "--test",
        help="Path to test logs directory"
    ),
    train_marker: str = typer.Option("Training Done!", help="Marker string for completed training"),
    test_marker: str = typer.Option("Confusion Matrix saved as:", help="Marker string for completed test")
):
    """
    Check training and test logs for completion markers.
    """
    print("📌 Checking training logs...")
    check_training_logs(train_path, train_marker)

    print("\n📌 Checking test logs...")
    check_test_logs(test_path, test_marker)


@contextmanager
def _tee_stdout(log_path: Optional[str]):
    """Simple tee: stdout + file (append)."""
    if not log_path:
        yield None
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fp:
        yield fp



@app.command("matrix-analyzer")
def matrix_analyzer_cmd(
    cm_root: Path = typer.Option(..., "--cm-root", help="Path to confusion_matrices root for ONE dataset"),
    db_path: Path = typer.Option(Path("results/db/experiment_logs.db"), "--db", help="SQLite DB path"),
    dataset: str = typer.Option(..., "--dataset", help="Dataset name: MNIST | GTSRB | GTSRB_RGB | LEGO"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Append console output to this file as well"),
):
    """
    Analyze confusion matrices and training logs, scoped to ONE dataset.
    """
    from src.analysis import matrix_analyzer as ma

    with _tee_stdout(str(log_file) if log_file else None) as fp:
        print(f"📝 Logging to: {log_file}" if log_file else "📝 Logging to: <stdout only>")
        print("🛠️ Rebuilding training_runs table...")
        ma.create_training_runs_table(str(db_path), log_fp=fp)
        ma.compute_training_times(str(db_path), log_fp=fp)

        print("📥 Collecting evaluation results...")
        ma.collect_and_store_results(str(cm_root), str(db_path), dataset=dataset, log_fp=fp)

        print("📊 Creating model summary table...")
        ma.create_model_summary_table(str(db_path), log_fp=fp)
        ma.compute_and_insert_model_summaries(str(db_path), dataset=dataset, log_fp=fp)

        print("📈 Querying best models (scoped)...")
        ma.query_best_models(str(db_path), dataset=dataset, log_fp=fp)

@app.command("preprocess")
def preprocess_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name: MNIST, GTSRB, LEGO"),
    base_dir: Path = typer.Option(Path("dataset"), help="Base directory containing datasets"),
    merged_dir_name: str = typer.Option("merged_datasets", help="Name of the merged output subdirectory"),
    max_tests: int = typer.Option(2000, help="Maximum number of generated test scenarios"),
    file_format: str = typer.Option("ubyte", "--format", "-f", help="File format: ubyte or npy")
):
    """
    Preprocess dataset: rotate images, merge datasets, generate scenarios.
    """
    dataset_config = {
        "MNIST": "dataset_mnist_non_rotated",
        "GTSRB": "dataset_GTSRB_non_rotated",
        "GTSRB_RGB": "dataset_GTSRB_RGB_non_rotated",
        "LEGO": "dataset_LEGO_non_rotated"
    }

    if dataset not in dataset_config:
        print(f"❌ Unsupported dataset: {dataset}")
        raise typer.Exit(code=1)

    if file_format.lower() == "npy":
        if not dataset.endswith("_RGB"):
            dataset_key = dataset + "_RGB"
            dataset_name = dataset_config.get(dataset_key, dataset_config[dataset])
        else:
            dataset_key = dataset
            dataset_name = dataset_config[dataset]
    else:
        dataset_key = dataset
        dataset_name = dataset_config[dataset]

    run_pipeline(
        base_dir=os.path.join(base_dir, dataset_key),
        dataset_name=dataset_name,
        dataset_key=dataset_key,
        merged_dir_name=merged_dir_name,
        max_tests=max_tests,
        file_format=file_format.lower()
    )

@app.command("learning-curves")
def learning_curves_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name: MNIST, GTSRB, GTSRB_RGB, LEGO"),
    logs_dir: Path = typer.Option(
        Path("results/log_files_from_slave/logs"),
        "--logs-dir",
        help="Base logs directory (either the parent of json_* or a specific json_<DATASET> folder).",
    ),
    output_dir: Path = typer.Option(
        Path("results/plots"),
        "--output-dir",
        help="Output directory for learning curves.",
    ),
):
    """
    Generate learning curves (loss/accuracy vs epoch) from training logs.
    """
    generate_learning_curves(dataset_name=dataset.strip(), logs_base=logs_dir, output_base=output_dir)


@app.command("optuna-analyze")
def optuna_analyze_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name (MNIST, GTSRB, GTSRB_RGB, LEGO)"),
    optuna_logs: Path = typer.Option(..., "--optuna-logs", help="Path to Optuna logs directory (optuna_checked/logs)"),
    output_dir: Path = typer.Option("results", "--output-dir", help="Base output directory"),
    curves_only: bool = typer.Option(False, "--curves-only", help="Generate only learning curves"),
    heatmaps_only: bool = typer.Option(False, "--heatmaps-only", help="Generate only [TEST] heatmaps"),
):
    """
    Analyze Optuna logs: training curves (with last checkpoint marker) and 1-row [TEST] heatmaps.
    """
    if not heatmaps_only:
        generate_optuna_learning_curves(dataset_name=dataset, optuna_logs_dir=optuna_logs, output_base=output_dir)
    if not curves_only:
        generate_optuna_heatmaps(dataset_name=dataset, optuna_logs_dir=optuna_logs, output_base=output_dir)

@app.command("best-models")
def best_models_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name: MNIST, GTSRB, GTSRB_RGB, LEGO"),
    results_dir: Path = typer.Option(
        Path("results/heatmaps"),
        "--results-dir",
        help="Folder containing the accuracy_matrix_*.csv made by 'analyze'.",
    ),
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        help="Optional custom output directory (default: <results>/<DATASET>/summary).",
    ),
):
    """
    Rank models for a dataset using generated accuracy matrices and save CSV summaries.
    """
    summarize_best_models(dataset_name=dataset.strip(), results_root=results_dir, out_dir=out_dir)


# @app.command()
# def prepare_dataset(
#     dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset to prepare: GTSRB, GTSRB_RGB, LEGO")
# ):
#     """Prepare a dataset from raw files."""
#
#     dataset = dataset.upper()
#     if dataset == "GTSRB":
#         gtsrb.main()
#     elif dataset == "GTSRB_RGB":
#         gtsrb_rgb.main()
#     elif dataset == "LEGO":
#         lego.main()
#     else:
#         typer.echo(f"❌ Unknown dataset: {dataset}")
#         raise typer.Exit(code=1)


if __name__ == "__main__":
    app()