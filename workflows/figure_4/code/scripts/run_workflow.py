#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if len(sys.argv) == 1:
    sys.argv.extend(["--config", str(ROOT / "config.yaml")])

from publication_analysis.workflow import main

if __name__ == "__main__":
    main()
