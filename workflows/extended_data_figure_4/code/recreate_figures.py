#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from package_utils import finish_run
import workflow


def main() -> None:
    started = time.time()
    tables = WORKFLOW_ROOT / "data" / "figure_tables"
    workflow.ensure_dirs()
    outputs = {
        "barcode_proportions_d05": workflow.plot_d05(),
        "barcode_proportions_d06": workflow.plot_d06(),
        "flow_panels": workflow.convert_flow_pdfs(),
    }
    finish_run(
        WORKFLOW_ROOT,
        "Extended Data Figure 4",
        started,
        [
            tables / "PL47_D05_summary_edit.csv",
            tables / "PL47_D06_barcode_hit_table.csv",
            tables / "flow_panel_sources",
        ],
        details={"figures": outputs},
    )


if __name__ == "__main__":
    main()
