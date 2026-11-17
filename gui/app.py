"""Launcher entry point for GUI."""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Execute root app.py
root_app = ROOT_DIR / "app.py"
with open(root_app, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(root_app), "exec")
    exec(code, globals())
