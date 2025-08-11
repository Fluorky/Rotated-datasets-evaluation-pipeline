import typer
from src.analysis.learning_matrix import process_dataset
# future: from src.db.handler import show_table
# future: from src.pipelines.rotation_pipeline import run_pipeline
# future: from src.utils.log_ingestor import ingest_logs
from src.analysis.log_ingestor import ingest_logs
from src.analysis.log_checker import check_test_logs, check_training_logs
from src.analysis import matrix_analyzer as ma
from src.pipelines.rotation_pipeline import run_pipeline
import typer
import src.datasets.gtsrb as gtsrb
import src.datasets.gtsrb_rgb as gtsrb_rgb
import src.datasets.lego as lego
import os

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


@app.command("matrix-analyzer")
def matrix_analyzer_cmd(
    cm_root: Path = typer.Option(
        Path(r"\\wsl.localhost\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs\json_MNIST\confusion_matrices"),
        "--cm-root",
        help="Path to root directory of confusion matrices"
    ),
    db_path: Path = typer.Option(
        Path("results/db/experiment_logs.db"),
        "--db",
        help="Path to SQLite database"
    )
):
    """
    Analyze confusion matrices and training logs, then update evaluation database.
    """


    print("🛠️ Rebuilding training_runs table...")
    ma.create_training_runs_table(db_path)
    ma.compute_training_times(db_path)

    print("📥 Collecting evaluation results...")
    ma.collect_and_store_results(str(cm_root), str(db_path))

    print("📊 Creating model summary table...")
    ma.create_model_summary_table(db_path)
    ma.compute_and_insert_model_summaries(db_path)

    print("📈 Querying best models...")
    ma.query_best_models(db_path)

@app.command("preprocess")
def preprocess_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name: MNIST, GTSRB, LEGO"),
    base_dir: Path = typer.Option(Path("dataset"), help="Base directory containing datasets"),
    merged_dir_name: str = typer.Option("merged_datasets", help="Name of the merged output subdirectory"),
    max_tests: int = typer.Option(2000, help="Maximum number of generated test scenarios")
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

    run_pipeline(
        base_dir=os.path.join(base_dir, dataset),
        dataset_name=dataset_config[dataset],
        dataset_key=dataset,
        merged_dir_name=merged_dir_name,
        max_tests=max_tests
    )

@app.command()
def prepare_dataset(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset to prepare: GTSRB, GTSRB_RGB, LEGO")
):
    """Prepare a dataset from raw files."""

    dataset = dataset.upper()
    if dataset == "GTSRB":
        gtsrb.main()
    elif dataset == "GTSRB_RGB":
        gtsrb_rgb.main()
    elif dataset == "LEGO":
        lego.main()
    else:
        typer.echo(f"❌ Unknown dataset: {dataset}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()