from pathlib import Path

wsl_log_dir = Path(r"\\wsl$\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs\json_4\train")

# The message indicating successful completion
SUCCESS_MARK = "Training Done!"

# List to hold logs that did not finish
incomplete_logs = []

# Scan all .txt files in the directory
for log_file in wsl_log_dir.glob("*.txt"):
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            if SUCCESS_MARK not in content:
                incomplete_logs.append(log_file.name)
    except Exception as e:
        print(f"Failed to read {log_file.name}: {e}")

# Report
if incomplete_logs:
    print("Log files missing 'Training Done!':")
    for file in incomplete_logs:
        print(f" - {file}")
else:
    print("All log files completed successfully.")


test_log_dir = Path(r"\\wsl$\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs\json_4\test")

# Marker indicating a successful test
SUCCESS_MARK = "Confusion Matrix saved as:"

# List to collect failed or incomplete test logs
incomplete_test_logs = []

# Iterate through all subdirectories
for subfolder in test_log_dir.iterdir():
    if subfolder.is_dir():
        for txt_file in subfolder.glob("*.txt"):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if SUCCESS_MARK not in content:
                        incomplete_test_logs.append(str(txt_file.relative_to(test_log_dir)))
            except Exception as e:
                print(f"Failed to read {txt_file.name}: {e}")

# Report results
if incomplete_test_logs:
    print("Test logs missing 'Confusion Matrix saved as:' line:")
    for log in incomplete_test_logs:
        print(f" - {log}")
else:
    print("All test logs completed successfully.")