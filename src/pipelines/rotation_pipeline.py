from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

from src.utils.handler import (
    generate_train_test_scenarios,   # agnostic – it walks dirs
    merge_npy_dirs,
    merge_ubyte_files,               # IDX
    npy_paths,
    rename_t10k_to_test,             # IDX/MNIST
    rotate_and_save_fixed_angle,     # IDX
    rotate_and_save_fixed_angle_npy,
    rotate_and_save_ranges,          # IDX
    rotate_and_save_ranges_npy,
)

FileFormat = Literal["ubyte", "npy"]
DEFAULT_SPLITS: tuple[str, str] = ("train", "test")
SUPPORTED_FILE_FORMATS = {"ubyte", "npy"}


def _normalize_file_format(file_format: str) -> FileFormat:
    value = file_format.strip().lower()
    if value not in SUPPORTED_FILE_FORMATS:
        allowed = ", ".join(sorted(SUPPORTED_FILE_FORMATS))
        raise ValueError(f"Unsupported file_format={file_format!r}. Expected one of: {allowed}")
    return value  # type: ignore[return-value]


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _idx_image_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"{split}-images-idx3-ubyte"


def _idx_label_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"{split}-labels-idx1-ubyte"


def _npy_image_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"{split}_images.npy"


def _npy_label_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"{split}_labels.npy"


def _expected_split_files(dataset_dir: Path, split: str, file_format: FileFormat) -> list[Path]:
    if file_format == "npy":
        return [_npy_image_path(dataset_dir, split), _npy_label_path(dataset_dir, split)]
    return [_idx_image_path(dataset_dir, split), _idx_label_path(dataset_dir, split)]


def _missing_files(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def _format_missing(paths: Sequence[Path]) -> str:
    return "\n".join(f"  - {path}" for path in paths)


def _output_state(expected_files: Sequence[Path]) -> str:
    """
    Return one of: missing, partial, complete.

    missing  -> no expected output files exist
    partial  -> some, but not all expected output files exist
    complete -> all expected output files exist
    """
    existing = [path.exists() for path in expected_files]
    if all(existing):
        return "complete"
    if any(existing):
        return "partial"
    return "missing"


def _guard_output(expected_files: Sequence[Path], label: str, overwrite: bool) -> bool:
    """
    Decide whether a step should run.

    Returns True when the caller should run the generation step.
    Returns False when complete output already exists and overwrite=False.
    Raises RuntimeError for partial outputs when overwrite=False, because continuing
    could mix old and new data.
    """
    if overwrite:
        return True

    state = _output_state(expected_files)
    if state == "complete":
        print(f"⏭️  Skipping existing output: {label}")
        return False
    if state == "partial":
        missing = _missing_files(expected_files)
        raise RuntimeError(
            f"Partial output already exists for {label}.\n"
            "Refusing to continue because this could mix old and new data.\n"
            "Delete the partial output or rerun with overwrite=True.\n"
            f"Missing files:\n{_format_missing(missing)}"
        )
    return True


def validate_pipeline_inputs(
    base_dir: str | Path,
    dataset_name: str,
    splits: Sequence[str] = DEFAULT_SPLITS,
    file_format: str = "ubyte",
) -> None:
    """
    Fail early before expensive rotations/merges start.

    Checks:
      - base_dir exists
      - dataset directory exists
      - file_format is supported
      - expected split image/label files exist
    """
    fmt = _normalize_file_format(file_format)
    base_path = _as_path(base_dir)
    dataset_dir = base_path / dataset_name

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")
    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    missing: list[Path] = []
    for split in splits:
        missing.extend(_missing_files(_expected_split_files(dataset_dir, split, fmt)))

    if missing:
        raise FileNotFoundError(
            f"Missing required {fmt} dataset files for dataset={dataset_name!r}:\n"
            f"{_format_missing(missing)}"
        )


def _validate_merge_sources(paths: Sequence[Path], merge_name: str) -> None:
    missing = [path for path in paths if not path.exists() or not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Cannot create merge preset {merge_name!r}; missing source directories:\n"
            f"{_format_missing(missing)}"
        )


def _fixed_angle_expected_outputs(
    base_dir: Path,
    angle: int,
    split: str,
    file_format: FileFormat,
) -> list[Path]:
    out_dir = base_dir / f"rotated-{angle}"
    if file_format == "npy":
        return [_npy_image_path(out_dir, split), _npy_label_path(out_dir, split)]
    return [_idx_image_path(out_dir, split)]


def _range_expected_outputs(
    base_dir: Path,
    ranges: Sequence[tuple[int, int]],
    split: str,
    file_format: FileFormat,
) -> list[Path]:
    expected: list[Path] = []
    for start, end in ranges:
        out_dir = base_dir / f"rotated-{start}-{end}"
        if file_format == "npy":
            expected.extend([_npy_image_path(out_dir, split), _npy_label_path(out_dir, split)])
        else:
            expected.append(_idx_image_path(out_dir, split))
    return expected


def _merge_expected_outputs(
    out_dir: Path,
    splits: Sequence[str],
    file_format: FileFormat,
) -> list[Path]:
    expected: list[Path] = []
    for split in splits:
        expected.extend(_expected_split_files(out_dir, split, file_format))
    return expected


def rotate_fixed_angles(
    base_dir: str | Path,
    dataset_name: str,
    angles: Sequence[int],
    splits: Sequence[str],
    file_format: str = "ubyte",
    overwrite: bool = False,
) -> None:
    """Rotate dataset by fixed angles for all splits."""
    base_path = _as_path(base_dir)
    fmt = _normalize_file_format(file_format)

    if fmt == "npy":
        for angle in angles:
            out_dir = base_path / f"rotated-{angle}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for split in splits:
                expected = _fixed_angle_expected_outputs(base_path, angle, split, fmt)
                if not _guard_output(expected, f"rotated-{angle}/{split} ({fmt})", overwrite):
                    continue

                images_in, labels_in = npy_paths(str(base_path), dataset_name, split)
                rotate_and_save_fixed_angle_npy(images_in, labels_in, str(out_dir), split, angle)
        return

    suffix = "-images-idx3-ubyte"
    for angle in angles:
        out_dir = base_path / f"rotated-{angle}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            expected = _fixed_angle_expected_outputs(base_path, angle, split, fmt)
            if not _guard_output(expected, f"rotated-{angle}/{split} ({fmt})", overwrite):
                continue

            input_file = base_path / dataset_name / f"{split}{suffix}"
            output_file = out_dir / f"{split}{suffix}"
            rotate_and_save_fixed_angle(str(input_file), str(output_file), angle)


def rotate_angle_ranges(
    base_dir: str | Path,
    dataset_name: str,
    ranges: Sequence[tuple[int, int]],
    splits: Sequence[str],
    file_format: str = "ubyte",
    overwrite: bool = False,
) -> None:
    """Rotate dataset across ranges of angles for all splits."""
    base_path = _as_path(base_dir)
    fmt = _normalize_file_format(file_format)

    if fmt == "npy":
        for split in splits:
            expected = _range_expected_outputs(base_path, ranges, split, fmt)
            if not _guard_output(expected, f"angle ranges/{split} ({fmt})", overwrite):
                continue

            images_in, labels_in = npy_paths(str(base_path), dataset_name, split)
            rotate_and_save_ranges_npy(
                images_in,
                labels_in,
                base_out=str(base_path),
                split=split,
                ranges=list(ranges),
            )
        return

    suffix = "-images-idx3-ubyte"
    for split in splits:
        expected = _range_expected_outputs(base_path, ranges, split, fmt)
        if not _guard_output(expected, f"angle ranges/{split} ({fmt})", overwrite):
            continue

        input_file = base_path / dataset_name / f"{split}{suffix}"
        rotate_and_save_ranges(str(input_file), str(base_path), list(ranges), split, suffix=suffix)


def predefined_merges(
    base_dir: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    angle_ranges: Sequence[tuple[int, int]],
    file_format: str = "ubyte",
    overwrite: bool = False,
) -> None:
    """Merge rotated datasets into defined presets."""
    base_path = _as_path(base_dir)
    output_path = _as_path(output_dir)
    fmt = _normalize_file_format(file_format)

    def d(angle: int) -> Path:
        return base_path / f"rotated-{angle}"

    def dr(start: int, end: int) -> Path:
        return base_path / f"rotated-{start}-{end}"

    def base() -> Path:
        return base_path / dataset_name

    fixed_30 = list(range(30, 360, 30))
    fixed_45 = list(range(45, 360, 45))

    def range_paths(ranges: Sequence[tuple[int, int]]) -> list[Path]:
        return [dr(start, end) for start, end in ranges]

    merge_configs: dict[str, list[Path]] = {
        "merged_fixed_30": [d(angle) for angle in fixed_30],
        "merged_fixed_45": [d(angle) for angle in fixed_45],
        "merged_fixed_all": [d(angle) for angle in sorted(set(fixed_30 + fixed_45))],
        "merged_range_0_90": range_paths(angle_ranges[:3]),
        "merged_range_90_180": range_paths(angle_ranges[3:6]),
        "merged_range_180_270": range_paths(angle_ranges[6:9]),
        "merged_range_270_360": range_paths(angle_ranges[9:]),
        "merged_range_0_180": range_paths(angle_ranges[:6]),
        "merged_range_180_360": range_paths(angle_ranges[6:]),
        "merged_range_full_0_360": range_paths(angle_ranges),
        "merged_range_full_0_360_plus_non_rotated": [base()] + range_paths(angle_ranges),
        "merged_fixed_30_plus_non_rotated": [base()] + [d(angle) for angle in fixed_30],
        "merged_fixed_45_plus_non_rotated": [base()] + [d(angle) for angle in fixed_45],
        "merged_range_0_180_plus_non_rotated": [base()] + range_paths(angle_ranges[:6]),
        "merged_range_180_360_plus_non_rotated": [base()] + range_paths(angle_ranges[6:]),
    }

    output_path.mkdir(parents=True, exist_ok=True)

    for name, paths in merge_configs.items():
        print(f"\n📁 Merging preset: {name}")
        for path in paths:
            print(f"  - {path}")

        _validate_merge_sources(paths, name)

        out_dir = output_path / name
        expected = _merge_expected_outputs(out_dir, DEFAULT_SPLITS, fmt)
        if not _guard_output(expected, f"merge preset {name} ({fmt})", overwrite):
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        if fmt == "npy":
            merge_npy_dirs([str(path) for path in paths], str(out_dir), splits=list(DEFAULT_SPLITS))
        else:
            merge_ubyte_files([str(path) for path in paths], str(out_dir))


def run_pipeline(
    base_dir: str | Path,
    dataset_name: str,
    dataset_key: str,
    merged_dir_name: str = "merged_datasets",
    max_tests: int = 2000,
    file_format: str = "ubyte",
    overwrite: bool = False,
) -> None:
    """
    Run the full rotation and merging pipeline for a given dataset and file format.

    By default the pipeline is idempotent: existing complete outputs are skipped,
    while partial outputs stop the run with a clear error. Pass overwrite=True to
    intentionally regenerate outputs.
    """
    fmt = _normalize_file_format(file_format)
    base_path = _as_path(base_dir)
    splits = list(DEFAULT_SPLITS)
    angle_ranges = [(i, i + 30) for i in range(0, 360, 30)]
    fixed_angles = sorted(set(range(30, 360, 30)).union(range(45, 360, 45)))

    # Only rename MNIST t10k -> test for IDX flows. This is a cheap preparation
    # step and is done before validation so validation can require test-* files.
    if dataset_key.upper().startswith("MNIST") and fmt != "npy":
        rename_t10k_to_test(str(base_path / dataset_name))

    print("🔎 Validating pipeline inputs...")
    validate_pipeline_inputs(base_path, dataset_name, splits=splits, file_format=fmt)

    merged_dir = base_path / merged_dir_name
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("🌀 Rotating fixed angles...")
    rotate_fixed_angles(base_path, dataset_name, fixed_angles, splits, fmt, overwrite=overwrite)

    print("🌀 Rotating angle ranges...")
    rotate_angle_ranges(base_path, dataset_name, angle_ranges, splits, fmt, overwrite=overwrite)

    print("🔧 Running predefined merges...")
    predefined_merges(base_path, dataset_name, merged_dir, angle_ranges, fmt, overwrite=overwrite)

    json_out_name = f"train_test_scenarios_{dataset_name.replace('dataset_', '')}.json"
    output_json_path = Path("configs") / "scenarios" / json_out_name
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if output_json_path.exists() and not overwrite:
        print(f"⏭️  Skipping existing train-test JSON: {output_json_path}")
    else:
        print("🧪 Generating train-test JSON...")
        generate_train_test_scenarios(
            merged_datasets_dir=str(base_path),
            output_json_path=str(output_json_path),
            max_tests=max_tests,
        )

    print(f"✅ All preprocessing completed. JSON path: {output_json_path}")
