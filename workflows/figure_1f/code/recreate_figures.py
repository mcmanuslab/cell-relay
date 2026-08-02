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
    dot_table = tables / "barcode_log2FC_dotplot_J_vs_K_table.csv"
    dot_groups = tables / "barcode_log2FC_dotplot_J_vs_K_groups.csv"
    violin_table = tables / "pmhc_log2FC_violin_by_sample_table.csv"
    violin_groups = tables / "pmhc_log2FC_violin_by_sample_groups.csv"
    violin_summary = tables / "pmhc_pMHC_summary_J_vs_K.csv"

    dot_outputs = workflow.plot_dotplot(pd.read_csv(dot_table), pd.read_csv(dot_groups))
    violin_outputs = workflow.plot_violin(
        pd.read_csv(violin_table),
        pd.read_csv(violin_groups),
        pd.read_csv(violin_summary),
    )
    finish_run(
        WORKFLOW_ROOT,
        "Figure 1F",
        started,
        [dot_table, dot_groups, violin_table, violin_groups, violin_summary],
        details={"dotplot": dot_outputs, "violin": violin_outputs},
    )


if __name__ == "__main__":
    main()
