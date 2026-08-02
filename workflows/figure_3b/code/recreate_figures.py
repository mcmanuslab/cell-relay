#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from package_utils import finish_run
import workflow


def main() -> None:
    started = time.time()
    table = (
        WORKFLOW_ROOT
        / "data"
        / "figure_tables"
        / "lfc_NT400-R1-1E4X_vs_NT400-R2-1E4X_table.csv"
    )
    outputs = workflow.plot_lfc_scatter(
        pd.read_csv(table), workflow.PLOT_SAMPLE_X, workflow.PLOT_SAMPLE_Y
    )
    finish_run(WORKFLOW_ROOT, "Figure 3B", started, [table], details={"figure": outputs})


if __name__ == "__main__":
    main()
