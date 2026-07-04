import pytest

from src.analysis.matrix_analyzer import (
    _angle_center_from_name,
    _bin_delta,
    _center_deg,
    _delta_deg,
    _interval_from_token,
    _is_train_like,
    _model_belongs_to_dataset,
)


def test_interval_parsing_for_known_dataset_variant_names():
    assert _interval_from_token("dataset_MNIST_non_rotated") == (0.0, 0.0)
    assert _interval_from_token("rotated-30") == (30.0, 30.0)
    assert _interval_from_token("rotated-30-60") == (30.0, 60.0)
    assert _interval_from_token("range_90_120") == (90.0, 120.0)
    assert _interval_from_token("range-180-210") == (180.0, 210.0)
    assert _interval_from_token("merged_range_full_0_360") == (0.0, 360.0)


def test_angle_centers_are_computed_from_intervals():
    assert _center_deg((30.0, 60.0)) == pytest.approx(45.0)
    assert _center_deg((330.0, 30.0)) == pytest.approx(0.0)
    assert _angle_center_from_name("range_90_120") == pytest.approx(105.0)


def test_delta_uses_shortest_circular_distance():
    assert _delta_deg("rotated-30", "rotated-90") == pytest.approx(60.0)
    assert _delta_deg("rotated-330", "rotated-30") == pytest.approx(60.0)
    assert _delta_deg("rotated-30-60", "rotated-90-120") == pytest.approx(60.0)
    assert _delta_deg("dataset_MNIST_non_rotated", "rotated-180") == pytest.approx(180.0)


def test_delta_returns_none_for_unparseable_names():
    assert _delta_deg("model_without_angle", "rotated-30") is None
    assert _delta_deg("rotated-30", "test_without_angle") is None


def test_bin_delta_rounds_to_step_and_caps_at_180():
    assert _bin_delta(0.0, step=15) == 0
    assert _bin_delta(7.0, step=15) == 0
    assert _bin_delta(8.0, step=15) == 15
    assert _bin_delta(181.0, step=15) == 180
    assert _bin_delta(300.0, step=15) == 180


def test_dataset_scoping_handles_gtsrb_vs_gtsrb_rgb():
    assert _model_belongs_to_dataset("GTSRB-Model", "GTSRB") is True
    assert _model_belongs_to_dataset("GTSRB_RGB-Model", "GTSRB") is False
    assert _model_belongs_to_dataset("GTSRB-RGB-Model", "GTSRB") is False
    assert _model_belongs_to_dataset("GTSRB_RGB-Model", "GTSRB_RGB") is True
    assert _model_belongs_to_dataset("GTSRB-RGB-Model", "GTSRB_RGB") is True


def test_train_like_detection_matches_baseline_and_plus_non_rotated_cases():
    assert _is_train_like("dataset_MNIST_non_rotated") is True
    assert _is_train_like("merged_range_full_0_360_plus_non_rotated") is True
    assert _is_train_like("rotated-90") is False
    assert _is_train_like("range_90_120") is False
