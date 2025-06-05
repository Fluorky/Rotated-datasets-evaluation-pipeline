import argparse
import subprocess
import os
import sys

# Source path inside WSL
WSL_SOURCE_PATH = "/home/testhub/CyCNN/CyCNN-master/cycnn/launcher.py"
# Target on Windows (slave/retrieved_launcher.py)
LOCAL_TARGET_PATH = os.path.abspath("slave/retrieved_launcher.py")


def to_wsl_path(win_path):
    drive, rest = os.path.splitdrive(win_path)
    drive_letter = drive[0].lower()
    rest = rest.replace("\\", "/")
    return f"/mnt/{drive_letter}{rest}"


def fetch_from_wsl():
    print(f"Fetching {WSL_SOURCE_PATH} from WSL to {LOCAL_TARGET_PATH}")
    wsl_target_path = to_wsl_path(LOCAL_TARGET_PATH)

    try:
        subprocess.run(
            ["wsl", "-d", "Ubuntu", "sh", "-c", f"cp '{WSL_SOURCE_PATH}' '{wsl_target_path}'"],
            check=True
        )
        print("✅ Retrieval successful.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Retrieval failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Deploy or fetch launcher.py")
    parser.add_argument("--get-current", action="store_true", help="Fetch current launcher.py from WSL")

    args = parser.parse_args()

    if args.get_current:
        fetch_from_wsl()
    else:
        print("No action specified. Use --get-current to fetch launcher.py.")


if __name__ == "__main__":
    main()
