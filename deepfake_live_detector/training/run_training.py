"""One-command training orchestrator.
Run from project root:
    python training/run_training.py              # Train both models
    python training/run_training.py --mode image  # Train image only
    python training/run_training.py --mode audio  # Train audio only
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.train import *

if __name__ == "__main__":
    # Re-use the unified train.py main
    import runpy
    sys.argv[0] = str(ROOT / "training" / "train.py")
    runpy.run_path(str(ROOT / "training" / "train.py"), run_name="__main__")
