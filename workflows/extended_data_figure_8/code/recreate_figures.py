#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "extended_data_figure8_github_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from package_utils import finish_run
import workflow


def render_mfi(table_path: Path) -> dict[str, str | int]:
    plot_df = pd.read_csv(table_path)
    labels = plot_df["Gene"].fillna(plot_df["Sample"])
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(plot_df)), 5), dpi=300)
    ax.bar(x, plot_df["A2_MFI"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Gene")
    ax.set_ylabel("A2_MFI")
    ax.set_title("A2 MFI by Gene")
    fig.tight_layout()
    out_dir = WORKFLOW_ROOT / "outputs" / "figures" / "flow"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "A2_MFI_by_Gene.pdf"
    png_path = out_dir / "A2_MFI_by_Gene.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return {"pdf": str(pdf_path.relative_to(WORKFLOW_ROOT)), "png": str(png_path.relative_to(WORKFLOW_ROOT)), "rows": len(plot_df)}


def main() -> None:
    started = time.time()
    tables = WORKFLOW_ROOT / "data" / "figure_tables"
    volcano_dir = tables / "rnaseq_volcano"
    sample_map_path = tables / "rnaseq_counts" / "JTRWM7_sample_map.tsv"
    mfi_path = tables / "A2_MFI_by_Gene_table.csv"
    histogram_path = tables / "flow_panel_sources" / "Histogram.svg"
    workflow.ensure_dirs()

    outputs: dict[str, object] = {"mfi": render_mfi(mfi_path)}
    outputs["histogram"] = workflow.render_svg_subset(
        histogram_path,
        WORKFLOW_ROOT / "outputs" / "figures" / "flow" / "Histogram.pdf",
        WORKFLOW_ROOT / "outputs" / "figures" / "flow" / "Histogram.png",
    )
    sample_map = pd.read_csv(sample_map_path, sep="\t")
    comparisons = [
        ("Index_4_vs_3", [4], [3]),
        ("Index_6_vs_5", [6], [5]),
        ("Index_8_vs_7", [8], [7]),
        ("Index_10_vs_9", [10], [9]),
    ]
    volcano_outputs = {}
    input_paths = [mfi_path, histogram_path, sample_map_path]
    for name, reference, test in comparisons:
        table_path = volcano_dir / f"{name}.tsv"
        labels_path = volcano_dir / f"{name}_labels.tsv"
        volcano_outputs[name] = workflow.plot_volcano_from_table(
            pd.read_csv(table_path, sep="\t"),
            pd.read_csv(labels_path, sep="\t"),
            sample_map,
            name,
            reference,
            test,
        )
        input_paths.extend([table_path, labels_path])
    outputs["volcano"] = volcano_outputs
    finish_run(
        WORKFLOW_ROOT,
        "Extended Data Figure 8",
        started,
        input_paths,
        details={"figures": outputs},
    )


if __name__ == "__main__":
    main()
