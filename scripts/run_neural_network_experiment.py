"""Run the minimal one-hidden-layer neural-network experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.neural_network_experiment import main

if __name__ == "__main__":
    main()
