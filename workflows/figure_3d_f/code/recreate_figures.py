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
    tables = WORKFLOW_ROOT / "data" / "figure_tables"
    ot_path = tables / "OT_plot_table.csv"
    wt_path = tables / "WT_plot_table.csv"
    ifn_path = tables / "IFN_vs_OT_plot_table.csv"
    outputs = {
        "OT": workflow.plot_normalized_lfc(pd.read_csv(ot_path), "OT"),
        "WT": workflow.plot_normalized_lfc(pd.read_csv(wt_path), "WT"),
        "IFN_vs_OT": workflow.plot_ifn_vs_ot(pd.read_csv(ifn_path)),
    }
    finish_run(
        WORKFLOW_ROOT,
        "Figure 3D-F",
        started,
        [ot_path, wt_path, ifn_path],
        details={"figures": outputs},
    )


if __name__ == "__main__":
    main()
