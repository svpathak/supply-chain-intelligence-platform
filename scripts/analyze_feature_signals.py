"""CLI entry point for validation-only feature signal analysis."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ml.feature_signals import main

if __name__ == "__main__":
    main()
