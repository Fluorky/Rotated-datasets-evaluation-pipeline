import sqlite3
import pytest
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

def _create_training_runs_db(db_path, rows):
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
    conn.executemany(
        """
        INSERT INTO training_runs (model, log_file, total_train_time)
        VALUES (?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _write_cm(cm_root, model, test_case, cm):
    target = cm_root / model / test_case
    target.mkdir(parents=True)
    np.save(target / "confusion_matrix.npy", np.asarray(cm))


def test_collect_and_store_results_stores_dataset_metric_and_time_fields(tmp_path):
    cm_root = tmp_path / "cm"
    db_path = tmp_path / "results.db"

    _write_cm(
        cm_root,
        model="MNIST-TestModel",
        test_case="dataset_MNIST_non_rotated",
        cm=np.array([[8, 2], [1, 9]]),
    )
    _create_training_runs_db(
        db_path,
        [("MNIST-TestModel", "MNIST-TestModel_train.txt", 10.0)],
    )

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

    assert row[0] == "MNIST-TestModel"
    assert row[1] == "dataset_MNIST_non_rotated"
    assert row[2] == pytest.approx(17 / 20)
    assert row[3] == pytest.approx(10.0)
    assert row[4] == pytest.approx((17 / 20) / 10.0)
    assert row[5] == "MNIST"
    assert row[6] == "micro"


def test_collect_and_store_results_keeps_micro_and_macro_separate(tmp_path):
    cm_root = tmp_path / "cm"
    db_path = tmp_path / "results.db"

    # micro = 90 / 101, macro = (90/100 + 0/1) / 2 = 0.45
    _write_cm(
        cm_root,
        model="MNIST-TestModel",
        test_case="dataset_MNIST_non_rotated",
        cm=np.array([[90, 10], [1, 0]]),
    )
    _create_training_runs_db(
        db_path,
        [("MNIST-TestModel", "MNIST-TestModel_train.txt", 10.0)],
    )

    collect_and_store_results(str(cm_root), str(db_path), dataset="MNIST", metric="micro")
    collect_and_store_results(str(cm_root), str(db_path), dataset="MNIST", metric="macro")

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
        ("macro", pytest.approx(0.45)),
        ("micro", pytest.approx(90 / 101)),
    ]


def test_clear_dataset_deletes_only_selected_dataset_and_metric(tmp_path):
    cm_root = tmp_path / "cm"
    db_path = tmp_path / "results.db"

    _write_cm(cm_root, "MNIST-TestModel", "dataset_MNIST_non_rotated", [[8, 2], [1, 9]])
    _create_training_runs_db(
        db_path,
        [("MNIST-TestModel", "MNIST-TestModel_train.txt", 10.0)],
    )

    collect_and_store_results(str(cm_root), str(db_path), dataset="MNIST", metric="micro")
    collect_and_store_results(str(cm_root), str(db_path), dataset="MNIST", metric="macro")
    collect_and_store_results(
        str(cm_root),
        str(db_path),
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

    assert rows == [("macro", 1), ("micro", 1)]


def test_collect_and_store_results_does_not_mix_gtsrb_and_gtsrb_rgb(tmp_path):
    cm_root = tmp_path / "cm"
    db_path = tmp_path / "results.db"

    _write_cm(cm_root, "GTSRB-Model", "dataset_GTSRB_non_rotated", [[5, 0], [0, 5]])
    _write_cm(cm_root, "GTSRB_RGB-Model", "dataset_GTSRB_RGB_non_rotated", [[1, 4], [0, 5]])
    _create_training_runs_db(
        db_path,
        [
            ("GTSRB-Model", "GTSRB-Model_train.txt", 10.0),
            ("GTSRB_RGB-Model", "GTSRB_RGB-Model_train.txt", 10.0),
        ],
    )

    collect_and_store_results(str(cm_root), str(db_path), dataset="GTSRB", metric="micro")

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT model, dataset FROM evaluations").fetchall()
    conn.close()

    assert rows == [("GTSRB-Model", "GTSRB")]
