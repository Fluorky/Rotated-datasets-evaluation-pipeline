import numpy as np
import pytest

from src.analysis.matrix_analyzer import (
    _macro_acc_from_cm,
    _micro_acc_from_cm,
    _model_belongs_to_dataset,
    _normalize_metric,
    _safe_load_cm,
)


def test_micro_accuracy_from_confusion_matrix():
    cm = np.array([[8, 2], [1, 9]])

    assert _micro_acc_from_cm(cm) == 17 / 20


def test_macro_accuracy_from_confusion_matrix():
    cm = np.array([
        [8, 2],   # class 0 accuracy: 8 / 10 = 0.8
        [1, 9],   # class 1 accuracy: 9 / 10 = 0.9
    ])

    assert _macro_acc_from_cm(cm) == pytest.approx(0.85)


def test_macro_accuracy_ignores_classes_with_zero_support():
    cm = np.array([
        [8, 2],
        [0, 0],   # no samples for this class
    ])

    assert _macro_acc_from_cm(cm) == 0.8


def test_dataset_scoping_does_not_mix_gtsrb_and_gtsrb_rgb():
    assert _model_belongs_to_dataset("GTSRB-CyVGG19", "GTSRB") is True
    assert _model_belongs_to_dataset("GTSRB_RGB-CyVGG19", "GTSRB") is False
    assert _model_belongs_to_dataset("GTSRB-RGB-CyVGG19", "GTSRB") is False
    assert _model_belongs_to_dataset("GTSRB_RGB-CyVGG19", "GTSRB_RGB") is True
    assert _model_belongs_to_dataset("GTSRB-RGB-CyVGG19", "GTSRB_RGB") is True


def test_normalize_metric_accepts_only_micro_or_macro():
    assert _normalize_metric("MICRO") == "micro"
    assert _normalize_metric(" macro ") == "macro"

    try:
        _normalize_metric("weighted")
    except ValueError as exc:
        assert "Unsupported metric" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported metric")


def test_safe_load_cm_accepts_valid_square_numeric_matrix(tmp_path):
    path = tmp_path / "confusion_matrix.npy"
    expected = np.array([[1, 2], [3, 4]])
    np.save(path, expected)

    loaded = _safe_load_cm(str(path))

    assert loaded is not None
    np.testing.assert_array_equal(loaded, expected)


def test_safe_load_cm_rejects_non_square_matrix(tmp_path):
    path = tmp_path / "bad_confusion_matrix.npy"
    np.save(path, np.array([[1, 2, 3], [4, 5, 6]]))

    assert _safe_load_cm(str(path)) is None


def test_safe_load_cm_rejects_negative_values(tmp_path):
    path = tmp_path / "bad_confusion_matrix.npy"
    np.save(path, np.array([[1, -1], [0, 2]]))

    assert _safe_load_cm(str(path)) is None
