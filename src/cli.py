from pathlib import Path
from typing import Optional
import sqlite3

import typer

from src.analysis.learning_matrix import process_dataset
from src.analysis.log_ingestor import ingest_logs
from src.analysis.log_checker import check_test_logs, check_training_logs
from src.analysis import matrix_analyzer as ma
from src.analysis.learning_curves import generate_learning_curves
from src.analysis.best_models import summarize_best_models
from src.analysis.optuna_analyzer import (
    generate_optuna_learning_curves,
    generate_optuna_heatmaps,
)
from src.pipelines.rotation_pipeline import run_pipeline


app = typer.Typer(
    help="CLI for generating rotated datasets and analyzing model evaluation results.",
    no_args_is_help=True,
)

SUPPORTED_DATASETS = {"MNIST", "GTSRB", "GTSRB_RGB", "LEGO"}
DATASET_CONFIG = {
    "MNIST": "dataset_mnist_non_rotated",
    "GTSRB": "dataset_GTSRB_non_rotated",
    "GTSRB_RGB": "dataset_GTSRB_RGB_non_rotated",
    "LEGO": "dataset_LEGO_non_rotated",
}


def _normalize_dataset(dataset: str) -> str:
    normalized = dataset.strip().upper().replace("-", "_")
    if normalized not in SUPPORTED_DATASETS:
        typer.echo(
            f"Unsupported dataset: {dataset}. Expected one of: {', '.join(sorted(SUPPORTED_DATASETS))}.",
            err=True,
        )
        raise typer.Exit(code=1)
    return normalized


def _normalize_metric(metric: str) -> str:
    normalized = metric.strip().lower()
    if normalized not in {"micro", "macro"}:
        typer.echo("Unsupported metric. Expected: micro or macro.", err=True)
        raise typer.Exit(code=1)
    return normalized


def _ensure_eval_indices(db_path: Path) -> None:
    """Create indexes used by matrix/result analysis commands."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_model   ON evaluations(model)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_dataset ON evaluations(dataset)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_metric  ON evaluations(metric)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_m_t     ON evaluations(model, test_case)")
        conn.commit()
    finally:
        conn.close()


@app.command("analyze")
def analyze_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset: MNIST | GTSRB | GTSRB_RGB | LEGO"),
    logs_dir: Path = typer.Option(
        Path("results/log_files_from_slave/logs"),
        "--logs-dir",
        help="Base logs directory.",
    ),
    output_dir: Path = typer.Option(
        Path("results/heatmaps"),
        "--output-dir",
        help="Output directory for heatmaps.",
    ),
) -> None:
    """Analyze logs and generate accuracy heatmaps."""
    dataset = _normalize_dataset(dataset)
    process_dataset(dataset_name=dataset, logs_base=logs_dir, output_base=output_dir)


@app.command("ingest")
def ingest_cmd(
    wsl_path: str = typer.Option(..., "--wsl-path", help="Source path to logs, usually a WSL path."),
    local_logs: Path = typer.Option(
        Path("results/log_files_from_slave/logs"),
        "--local-logs",
        help="Local logs directory used by this repository.",
    ),
    db_path: Path = typer.Option(
        Path("results/db/experiment_logs.db"),
        "--db",
        help="SQLite database path.",
    ),
    overwrite_logs: bool = typer.Option(False, "--overwrite-logs", help="Force re-sync from source logs."),
    overwrite_db: bool = typer.Option(False, "--overwrite-db", help="Overwrite existing DB entries."),
) -> None:
    """Sync logs from an external source and ingest them into the experiment database."""
    ingest_logs(
        wsl_source=wsl_path,
        local_logs_path=str(local_logs),
        db_path=str(db_path),
        overwrite_logs=overwrite_logs,
        overwrite_db=overwrite_db,
    )


@app.command("check-logs")
def check_logs_cmd(
    dataset: str = typer.Option("MNIST", "--dataset", "-d", help="Dataset: MNIST | GTSRB | GTSRB_RGB | LEGO"),
    logs_dir: Path = typer.Option(
        Path("results/log_files_from_slave/logs"),
        "--logs-dir",
        help="Base logs directory containing json_<DATASET>/train and json_<DATASET>/test.",
    ),
    train_path: Optional[Path] = typer.Option(
        None,
        "--train",
        help="Optional explicit path to training logs directory. Overrides --logs-dir/--dataset.",
    ),
    test_path: Optional[Path] = typer.Option(
        None,
        "--test",
        help="Optional explicit path to test logs directory. Overrides --logs-dir/--dataset.",
    ),
    train_marker: str = typer.Option("Training Done!", "--train-marker", help="Marker string for completed training."),
    test_marker: str = typer.Option(
        "Confusion Matrix saved as:",
        "--test-marker",
        help="Marker string for completed test.",
    ),
) -> None:
    """Check training and test logs for completion markers."""
    dataset = _normalize_dataset(dataset)
    dataset_logs_dir = logs_dir / f"json_{dataset}"
    train_path = train_path or dataset_logs_dir / "train"
    test_path = test_path or dataset_logs_dir / "test"

    typer.echo(f"Checking training logs: {train_path}")
    check_training_logs(train_path, train_marker)

    typer.echo(f"\nChecking test logs: {test_path}")
    check_test_logs(test_path, test_marker)


@app.command("matrix-analyzer")
def matrix_analyzer_cmd(
    cm_root: Path = typer.Option(
        ...,
        "--cm-root",
        help="Root of confusion matrices: <MODEL>/<TEST_CASE>/confusion_matrix.npy.",
    ),
    db_path: Path = typer.Option(
        Path("results/db/experiment_logs.db"),
        "--db",
        help="SQLite database path.",
    ),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset: MNIST | GTSRB | GTSRB_RGB | LEGO"),
    metric: str = typer.Option("micro", "--metric", help="Accuracy metric: micro | macro."),
    clear: bool = typer.Option(
        False,
        "--clear/--no-clear",
        help="Clear existing evaluation rows for this dataset before inserting new rows.",
    ),
    theta_step: int = typer.Option(15, "--theta-step", help="Angle bin width in degrees for delta-theta curves."),
    alpha: float = typer.Option(0.70, "--alpha", help="Balanced score weight: alpha*quality + (1-alpha)*efficiency."),
    top_n: int = typer.Option(50, "--top-n", help="Number of top models to print."),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Optional path for tee'd console output."),
    per_class_angles: bool = typer.Option(False, "--per-class-angles", help="Export per-class-vs-angle analysis."),
    by_delta: bool = typer.Option(False, "--by-delta", help="Use delta-theta instead of test angle for per-class analysis."),
) -> None:
    """Analyze confusion matrices and export per-dataset rankings and summaries."""
    dataset = _normalize_dataset(dataset)
    metric = _normalize_metric(metric)

    with ma.tee_to_file(str(log_file) if log_file else None):
        ma._log_header(dataset=dataset, alpha=alpha, metric=metric, theta_step=theta_step, top_n=top_n)

        ma.create_training_runs_table(str(db_path))
        ma.compute_training_times(str(db_path))

        if hasattr(ma, "_ensure_eval_extra_cols"):
            ma._ensure_eval_extra_cols(str(db_path))

        ma.collect_and_store_results(
            cm_root=str(cm_root),
            db_path=str(db_path),
            dataset=dataset,
            metric=metric,
            clear_dataset=clear,
        )

        ma.create_model_summary_table(str(db_path))
        ma.compute_and_insert_model_summaries(str(db_path))
        _ensure_eval_indices(db_path)

        ma.query_best_models(
            db_path=str(db_path),
            dataset=dataset,
            alpha=alpha,
            top_n=top_n,
            theta_step=theta_step,
        )

        if per_class_angles:
            ma.export_per_class_vs_angle(
                cm_root=str(cm_root),
                dataset=dataset,
                theta_step=theta_step,
                by_delta=by_delta,
            )


@app.command("preprocess")
def preprocess_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset: MNIST | GTSRB | GTSRB_RGB | LEGO"),
    base_dir: Path = typer.Option(Path("dataset"), "--base-dir", help="Base directory containing dataset folders."),
    merged_dir_name: str = typer.Option("merged_datasets", "--merged-dir-name", help="Merged output subdirectory name."),
    max_tests: int = typer.Option(2000, "--max-tests", help="Maximum number of generated test scenarios."),
    file_format: str = typer.Option("ubyte", "--format", "-f", help="Dataset file format: ubyte | npy."),
) -> None:
    """Generate rotated dataset variants, merged datasets, and train/test scenarios."""
    dataset = _normalize_dataset(dataset)
    file_format = file_format.strip().lower()
    if file_format not in {"ubyte", "npy"}:
        typer.echo("Unsupported format. Expected: ubyte or npy.", err=True)
        raise typer.Exit(code=1)

    run_pipeline(
        base_dir=str(base_dir / dataset),
        dataset_name=DATASET_CONFIG[dataset],
        dataset_key=dataset,
        merged_dir_name=merged_dir_name,
        max_tests=max_tests,
        file_format=file_format,
    )


@app.command("learning-curves")
def learning_curves_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset: MNIST | GTSRB | GTSRB_RGB | LEGO"),
    logs_dir: Path = typer.Option(
        Path("results/log_files_from_slave/logs"),
        "--logs-dir",
        help="Base logs directory, either parent of json_* folders or a specific json_<DATASET> folder.",
    ),
    output_dir: Path = typer.Option(
        Path("results/plots"),
        "--output-dir",
        help="Output directory for learning curves.",
    ),
) -> None:
    """Generate learning curves from training logs."""
    dataset = _normalize_dataset(dataset)
    generate_learning_curves(dataset_name=dataset, logs_base=logs_dir, output_base=output_dir)


@app.command("optuna-analyze")
def optuna_analyze_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset: MNIST | GTSRB | GTSRB_RGB | LEGO"),
    optuna_logs: Path = typer.Option(..., "--optuna-logs", help="Path to Optuna logs directory."),
    output_dir: Path = typer.Option(Path("results"), "--output-dir", help="Base output directory."),
    curves_only: bool = typer.Option(False, "--curves-only", help="Generate only learning curves."),
    heatmaps_only: bool = typer.Option(False, "--heatmaps-only", help="Generate only test heatmaps."),
) -> None:
    """Analyze Optuna logs: training curves and test heatmaps."""
    dataset = _normalize_dataset(dataset)

    if curves_only and heatmaps_only:
        typer.echo("Use either --curves-only or --heatmaps-only, not both.", err=True)
        raise typer.Exit(code=1)

    if not heatmaps_only:
        generate_optuna_learning_curves(dataset_name=dataset, optuna_logs_dir=optuna_logs, output_base=output_dir)
    if not curves_only:
        generate_optuna_heatmaps(dataset_name=dataset, optuna_logs_dir=optuna_logs, output_base=output_dir)


@app.command("best-models")
def best_models_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset: MNIST | GTSRB | GTSRB_RGB | LEGO"),
    results_dir: Path = typer.Option(
        Path("results/heatmaps"),
        "--results-dir",
        help="Directory containing accuracy_matrix_*.csv files generated by the analyze command.",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        help="Optional custom output directory. Default: <results>/<DATASET>/summary.",
    ),
) -> None:
    """Rank models for a dataset and save CSV summaries."""
    dataset = _normalize_dataset(dataset)
    summarize_best_models(dataset_name=dataset, results_root=results_dir, out_dir=out_dir)


if __name__ == "__main__":
    app()
