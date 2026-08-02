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
    table = WORKFLOW_ROOT / "data" / "figure_tables" / "umi_coupling_stacked_bar_table.csv"
    outputs = workflow.plot_umi_coupling(pd.read_csv(table))
    finish_run(WORKFLOW_ROOT, "Figure 2I", started, [table], details={"figure": outputs})


if __name__ == "__main__":
    main()
