import sqlite3

import numpy as np

from src.analysis.matrix_analyzer import (
    collect_and_store_results,
    create_training_runs_table,
)


def test_collect_and_store_results_stores_dataset_and_metric(tmp_path):
    cm_root = tmp_path / "cm"
    model_dir = cm_root / "MNIST-TestModel" / "dataset_MNIST_non_rotated"
    model_dir.mkdir(parents=True)

    cm = np.array([[8, 2], [1, 9]])
    np.save(model_dir / "confusion_matrix.npy", cm)

    db_path = tmp_path / "results.db"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE training_runs (
            model TEXT PRIMARY KEY,
            log_file TEXT,
            total_train_time REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO training_runs (model, log_file, total_train_time)
        VALUES (?, ?, ?)
        """,
        ("MNIST-TestModel", "MNIST-TestModel_train.txt", 10.0),
    )
    conn.commit()
    conn.close()

    collect_and_store_results(
        cm_root=str(cm_root),
        db_path=str(db_path),
        dataset="MNIST",
        metric="micro",
        clear_dataset=False,
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT model, test_case, accuracy, train_time, perf_per_time, dataset, metric
        FROM evaluations
        """
    ).fetchone()
    conn.close()

    assert row == (
        "MNIST-TestModel",
        "dataset_MNIST_non_rotated",
        17 / 20,
        10.0,
        (17 / 20) / 10.0,
        "MNIST",
        "micro",
    )


def test_collect_and_store_results_keeps_micro_and_macro_separate(tmp_path):
    cm_root = tmp_path / "cm"
    model_dir = cm_root / "MNIST-TestModel" / "dataset_MNIST_non_rotated"
    model_dir.mkdir(parents=True)

    cm = np.array([[8, 2], [0, 0]])
    np.save(model_dir / "confusion_matrix.npy", cm)

    db_path = tmp_path / "results.db"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE training_runs (
            model TEXT PRIMARY KEY,
            log_file TEXT,
            total_train_time REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO training_runs (model, log_file, total_train_time)
        VALUES (?, ?, ?)
        """,
        ("MNIST-TestModel", "MNIST-TestModel_train.txt", 10.0),
    )
    conn.commit()
    conn.close()

    collect_and_store_results(
        cm_root=str(cm_root),
        db_path=str(db_path),
        dataset="MNIST",
        metric="micro",
        clear_dataset=False,
    )

    collect_and_store_results(
        cm_root=str(cm_root),
        db_path=str(db_path),
        dataset="MNIST",
        metric="macro",
        clear_dataset=False,
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT metric, accuracy
        FROM evaluations
        ORDER BY metric
        """
    ).fetchall()
    conn.close()

    assert rows == [
        ("macro", 0.8),
        ("micro", 0.8),
    ]


def test_clear_dataset_deletes_only_selected_metric(tmp_path):
    cm_root = tmp_path / "cm"
    model_dir = cm_root / "MNIST-TestModel" / "dataset_MNIST_non_rotated"
    model_dir.mkdir(parents=True)

    cm = np.array([[8, 2], [1, 9]])
    np.save(model_dir / "confusion_matrix.npy", cm)

    db_path = tmp_path / "results.db"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE training_runs (
            model TEXT PRIMARY KEY,
            log_file TEXT,
            total_train_time REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO training_runs (model, log_file, total_train_time)
        VALUES (?, ?, ?)
        """,
        ("MNIST-TestModel", "MNIST-TestModel_train.txt", 10.0),
    )
    conn.commit()
    conn.close()

    collect_and_store_results(
        cm_root=str(cm_root),
        db_path=str(db_path),
        dataset="MNIST",
        metric="micro",
        clear_dataset=False,
    )

    collect_and_store_results(
        cm_root=str(cm_root),
        db_path=str(db_path),
        dataset="MNIST",
        metric="macro",
        clear_dataset=False,
    )

    collect_and_store_results(
        cm_root=str(cm_root),
        db_path=str(db_path),
        dataset="MNIST",
        metric="micro",
        clear_dataset=True,
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT metric, COUNT(*)
        FROM evaluations
        GROUP BY metric
        ORDER BY metric
        """
    ).fetchall()
    conn.close()

    assert rows == [
        ("macro", 1),
        ("micro", 1),
    ]