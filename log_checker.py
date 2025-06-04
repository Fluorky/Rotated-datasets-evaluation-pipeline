from pathlib import Path

def check_training_logs(log_dir: Path, success_mark: str = "Training Done!"):
    """
    Checks training log files for the presence of a success marker.

    :param log_dir: Path to the directory containing training log files.
    :param success_mark: Marker indicating successful completion of training.
    """
    incomplete_logs = []

    for log_file in log_dir.glob("*.txt"):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
                if success_mark not in content:
                    incomplete_logs.append(log_file.name)
        except Exception as e:
            print(f"Failed to read {log_file.name}: {e}")

    if incomplete_logs:
        print("Training log files missing success marker:")
        for file in incomplete_logs:
            print(f" - {file}")
    else:
        print("All training log files completed successfully.")


def check_test_logs(test_dir: Path, success_mark: str = "Confusion Matrix saved as:"):
    """
    Checks test log files in subdirectories for the presence of a success marker.

    :param test_dir: Path to the directory containing subfolders with test log files.
    :param success_mark: Marker indicating successful completion of a test.
    """
    incomplete_logs = []

    for subfolder in test_dir.iterdir():
        if subfolder.is_dir():
            for txt_file in subfolder.glob("*.txt"):
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if success_mark not in content:
                            incomplete_logs.append(str(txt_file.relative_to(test_dir)))
                except Exception as e:
                    print(f"Failed to read {txt_file.name}: {e}")

    if incomplete_logs:
        print("Test log files missing success marker:")
        for log in incomplete_logs:
            print(f" - {log}")
    else:
        print("All test log files completed successfully.")



train_path = Path(r"\\wsl$\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs\json_4\train")
test_path = Path(r"\\wsl$\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs\json_4\test")

check_training_logs(train_path)
check_test_logs(test_path)
