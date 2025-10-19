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



# in src/cli.py
from src.analysis import matrix_analyzer as ma
from typing import Optional
from pathlib import Path
import typer
from src.analysis import matrix_analyzer as ma

@app.command("matrix-analyzer")
def matrix_analyzer_cmd(
    cm_root: Path = typer.Option(
        ...,
        "--cm-root",
        help="Root of confusion matrices: <MODEL>/<TEST_CASE>/confusion_matrix.npy",
    ),
    db_path: Path = typer.Option(
        Path("results/db/experiment_logs.db"),
        "--db",
        help="SQLite DB path",
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset", "-d",
        help="Dataset: MNIST | GTSRB | GTSRB_RGB | LEGO",
    ),
    metric: str = typer.Option(
        "micro",
        "--metric",
        help="Accuracy from confusion matrix: micro | macro",
        case_sensitive=False,
    ),
    clear: bool = typer.Option(
        True,
        "--clear/--no-clear",
        help="Clear existing rows for this dataset before inserting new ones",
    ),
    theta_step: int = typer.Option(
        15,
        "--theta-step",
        help="Angle bin width (degrees) for Δθ curves/AUCθ, e.g. 15",
    ),
    alpha: float = typer.Option(
        0.70,
        "--alpha",
        help="Balanced score α: score = α·norm(avg) + (1-α)·norm(avg_perf)",
    ),
    top_n: int = typer.Option(
        50,
        "--top-n",
        help="How many models to show in the tops",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Optional: write a tee'd console log to this file",
    ),
):
    """
    Analyze confusion matrices and produce per-dataset rankings and CSV exports.
    """

    # Small helper: create DB indices for faster queries on large runs.
    def _ensure_indices(dbp: Path):
        conn = ma.sqlite3.connect(str(dbp)); cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_model   ON evaluations(model)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_dataset ON evaluations(dataset)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_metric  ON evaluations(metric)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_m_t     ON evaluations(model, test_case)")
        conn.commit(); conn.close()

    with ma.tee_to_file(str(log_file) if log_file else None):
        ma._log_header(dataset=dataset, alpha=alpha, metric=metric, theta_step=theta_step, top_n=top_n)

        # Build/refresh training time table from training_logs (if present)
        ma.create_training_runs_table(str(db_path))
        ma.compute_training_times(str(db_path))

        # Ensure schema migration (adds columns: dataset, metric) **before** any UPDATEs
        ma._ensure_eval_extra_cols(str(db_path))

        # Scan CMs and insert enriched rows
        ma.collect_and_store_results(
            cm_root=str(cm_root),
            db_path=str(db_path),
            dataset=dataset,
            metric=metric,
            clear_dataset=clear,
        )

        # Optional legacy table
        ma.create_model_summary_table(str(db_path))
        ma.compute_and_insert_model_summaries(str(db_path))

        # Reports + CSV exports
        _ensure_indices(db_path)
        ma.query_best_models(
            db_path=str(db_path),
            dataset=dataset,
            alpha=alpha,
            top_n=top_n,
            theta_step=theta_step,
        )

        # # make metric visible for matrix_analyzer (export folder naming)
        # ma._CURRENT_METRIC = (metric or "micro").lower().strip()
        #
        # ma._log_header(dataset=dataset, alpha=alpha, metric=metric, theta_step=theta_step, top_n=top_n)
        # ma.create_training_runs_table(str(db_path))
        # ma.compute_training_times(str(db_path))
        #
        # ma.collect_and_store_results(
        #     cm_root=str(cm_root),
        #     db_path=str(db_path),
        #     dataset=dataset,
        #     metric=metric,
        #     clear_dataset=clear
        # )
        #
        # # pass metric so CSVs land in .../<dataset>/<micro|macro>/
        # ma.query_best_models(
        #     db_path=str(db_path),
        #     dataset=dataset,
        #     alpha=alpha,
        #     top_n=top_n,
        #     theta_step=theta_step,
        #     metric=metric,
        # )


#
# @app.command("matrix-analyzer")
# def matrix_analyzer_cmd(
#     cm_root: Path = typer.Option(..., "--cm-root", help="Path to root directory of confusion matrices"),
#     db_path: Path = typer.Option(..., "--db", help="Path to SQLite database"),
#     dataset: str = typer.Option(..., "--dataset", help="Dataset name: MNIST, GTSRB, GTSRB_RGB, LEGO"),
#     log_file: Path = typer.Option(None, "--log-file", help="Optional path to save console output"),
#     alpha: float = typer.Option(0.70, "--alpha", help="Balance weight for quality vs efficiency"),
#     metric: str = typer.Option("micro", "--metric", help="Accuracy metric: micro or macro"),
#     theta_step: int = typer.Option(15, "--theta-step", help="Angle bin in degrees (for Δθ curves)"),
#     top_n: int = typer.Option(50, "--top-n", help="How many items to show in rankings"),
#     clear: bool = typer.Option(False, "--clear", help="Clear existing eval rows for this dataset before inserting"),
# ):
#     """
#     Analyze confusion matrices for a single dataset and print/export rankings.
#     """
#     from src.analysis import matrix_analyzer as ma
#
#
#     with ma.tee_to_file(str(log_file) if log_file else None):
#         ma._log_header(dataset=dataset, alpha=alpha, metric=metric, theta_step=theta_step, top_n=top_n)
#         ma.create_training_runs_table(str(db_path))
#         ma.compute_training_times(str(db_path))
#         ma.collect_and_store_results(
#             cm_root=str(cm_root),
#             db_path=str(db_path),
#             dataset=dataset,
#             metric=metric,
#             clear_dataset=clear,
#         )
#
#         ma.create_model_summary_table(str(db_path))
#         ma.compute_and_insert_model_summaries(str(db_path))
#
#         ma.query_best_models(
#             db_path=str(db_path),
#             dataset=dataset,
#             alpha=alpha,
#             top_n=top_n,
#             theta_step=theta_step,
#         )

#
# @app.command("matrix-analyzer")
# def matrix_analyzer_cmd(
#     cm_root: Path = typer.Option(..., "--cm-root", help="Path to <json_DATASET>/confusion_matrices"),
#     db_path: Path = typer.Option(Path("results/db/experiment_logs.db"), "--db", help="SQLite DB path"),
#     dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name: MNIST | GTSRB | GTSRB_RGB | LEGO"),
#     log_file: Path = typer.Option(None, "--log-file", help="Optional log file (tee stdout)"),
#     alpha: float = typer.Option(0.70, "--alpha", help="Weight for balanced score (quality vs efficiency)"),
#     top_n: int = typer.Option(500, "--top", help="Top-N in printed rankings")
# ):
#     """
#     Analyze confusion matrices and training logs (per dataset). Populates DB and prints rankings.
#     """
#     with ma.tee_to_file(str(log_file) if log_file else None):
#         print("🛠️ Rebuilding training_runs table...")
#         ma.create_training_runs_table(str(db_path))
#         ma.compute_training_times(str(db_path))
#
#         print("📥 Collecting evaluation results...")
#         ma.collect_and_store_results(str(cm_root), str(db_path), dataset=dataset)
#
#         print("📈 Querying best models (per dataset)...")
#         ma.query_best_models(str(db_path), dataset=dataset, alpha=alpha, top_n=top_n)

@app.command("matrix-analyzer")
def matrix_analyzer_cmd(
    cm_root: Path = typer.Option(..., "--cm-root", help="Root of confusion matrices"),
    db_path: Path = typer.Option(Path("results/db/experiment_logs.db"), "--db", help="SQLite DB"),
    dataset: str = typer.Option(..., "--dataset", help="Dataset name"),
    metric: str = typer.Option("micro", "--metric", help="micro|macro"),
    clear: bool = typer.Option(False, "--clear", help="Clear previous rows for this dataset"),
    theta_step: int = typer.Option(15, "--theta-step", help="Angle bin (deg)"),
    alpha: float = typer.Option(0.70, "--alpha", help="Balanced score alpha"),
    top_n: int = typer.Option(50, "--top-n", help="How many to print"),
    log_file: Path = typer.Option(None, "--log-file", help="Log file"),
    per_class_angles: bool = typer.Option(False, "--per-class-angles", help="Export per-class vs angle"),
    by_delta: bool = typer.Option(False, "--by-delta", help="Use Δθ instead of test-angle"),
):
    from src.analysis import matrix_analyzer as ma

    with ma.tee_to_file(str(log_file) if log_file else None):
        ma._log_header(dataset=dataset, alpha=alpha, metric=metric, theta_step=theta_step, top_n=top_n)

        ma.create_training_runs_table(str(db_path))
        ma.compute_training_times(str(db_path))
        ma.collect_and_store_results(
            cm_root=str(cm_root),
            db_path=str(db_path),
            dataset=dataset,
            metric=metric,
            clear_dataset=clear,
        )
        ma.create_model_summary_table(str(db_path))
        ma.compute_and_insert_model_summaries(str(db_path))
        ma.query_best_models(
            db_path=str(db_path), dataset=dataset,
            alpha=alpha, top_n=top_n, theta_step=theta_step
        )

        if per_class_angles:
            ma.export_per_class_vs_angle(
                cm_root=str(cm_root),
                dataset=dataset,
                theta_step=theta_step,
                by_delta=by_delta,
            )


# @app.command("preprocess")
# def preprocess_cmd(
#     dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name: MNIST, GTSRB, LEGO"),
#     base_dir: Path = typer.Option(Path("dataset"), help="Base directory containing datasets"),
#     merged_dir_name: str = typer.Option("merged_datasets", help="Name of the merged output subdirectory"),
#     max_tests: int = typer.Option(2000, help="Maximum number of generated test scenarios"),
#     file_format: str = typer.Option("ubyte", "--format", "-f", help="File format: ubyte or npy")
# ):
#     """
#     Preprocess dataset: rotate images, merge datasets, generate scenarios.
#     """
#     dataset_config = {
#         "MNIST": "dataset_mnist_non_rotated",
#         "GTSRB": "dataset_GTSRB_non_rotated",
#         "GTSRB_RGB": "dataset_GTSRB_RGB_non_rotated",
#         "LEGO": "dataset_LEGO_non_rotated"
#     }
#
#     if dataset not in dataset_config:
#         print(f"❌ Unsupported dataset: {dataset}")
#         raise typer.Exit(code=1)
#
#     if file_format.lower() == "npy":
#         if not dataset.endswith("_RGB"):
#             dataset_key = dataset + "_RGB"
#             dataset_name = dataset_config.get(dataset_key, dataset_config[dataset])
#         else:
#             dataset_key = dataset
#             dataset_name = dataset_config[dataset]
#     else:
#         dataset_key = dataset
#         dataset_name = dataset_config[dataset]
#
#     run_pipeline(
#         base_dir=os.path.join(base_dir, dataset_key),
#         dataset_name=dataset_name,
#         dataset_key=dataset_key,
#         merged_dir_name=merged_dir_name,
#         max_tests=max_tests,
#         file_format=file_format.lower()
#     )

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