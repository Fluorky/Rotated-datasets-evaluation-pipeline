import shutil
import os

# Local path on Windows
source_path = os.path.abspath("slave/launcher.py")

# Adjust "Ubuntu" to match your actual WSL distro name
target_path = r"\\wsl.localhost\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\launcher.py"

try:
    shutil.copy(source_path, target_path)
    print("launcher.py was deployed to WSL via \\wsl$ path.")
except Exception as e:
    print("Deployment failed:", e)
