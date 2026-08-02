#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "figure5_github_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from package_utils import finish_run
import screen_analysis
import workflow


def draw_validation_heatmap(data: pd.DataFrame, title: str, name: str) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "validation_diverging", ["#5d8af7", "#f7f7f7", "#ed8590"], N=256
    )
    cell_size = 0.42
    fig_w = max(4.5, data.shape[1] * cell_size + 2.8)
    fig_h = max(6.0, data.shape[0] * cell_size + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    image = ax.imshow(
        data.to_numpy(), cmap=cmap, vmin=-2, vmax=2, interpolation="none", aspect="equal"
    )
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, fontsize=10)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index, fontsize=9)
    ax.set_title(title, fontsize=12, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("LFC", fontsize=10)
    fig.subplots_adjust(left=0.22, right=0.88, top=0.93, bottom=0.06)
    out_dir = WORKFLOW_ROOT / "outputs" / "figures" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.time()
    tables = WORKFLOW_ROOT / "data" / "figure_tables"
    figures = WORKFLOW_ROOT / "outputs" / "figures"
    summaries = WORKFLOW_ROOT / "outputs" / "summaries"
    for path in [
        figures / "barcode_stacks",
        figures / "decontamination",
        figures / "validation",
        figures / "umi_qc",
        figures / "screen_analysis",
        figures / "program_cards",
        summaries,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    dotplot = tables / "B_vs_C_dotplot_table.csv"
    workflow.plot_b_vs_c_dotplot(pd.read_csv(dotplot))

    sgrna_paths = []
    for label in ["B.sgrna_summary", "C.sgrna_summary"]:
        table_path = tables / f"{label}_barcode_stack_table.csv"
        workflow.plot_barcode_stack(table=pd.read_csv(table_path), label=label)
        sgrna_paths.append(table_path)

    decontamination = tables / "decontamination_lfc_table.csv"
    workflow.plot_decontam_density(pd.read_csv(decontamination))

    k562_path = tables / "validation_heatmap_K562_table.csv"
    t293_path = tables / "validation_heatmap_293T_table.csv"
    draw_validation_heatmap(pd.read_csv(k562_path, index_col=0), "K562", "validation_heatmap_K562")
    draw_validation_heatmap(pd.read_csv(t293_path, index_col=0), "293T", "validation_heatmap_293T")

    umi_paths = sorted(tables.glob("*__perID_UMI_counts.csv"))
    umi_summary = workflow.plot_umi_qc()

    relay_hits = tables / "relay_hits_regenerated.csv"
    card_summary = screen_analysis.plot_program_cards(pd.read_csv(relay_hits))

    finish_run(
        WORKFLOW_ROOT,
        "Figure 5",
        started,
        [dotplot, decontamination, k562_path, t293_path, relay_hits, *sgrna_paths, *umi_paths],
        details={
            "r_dotplot_engine": "R",
            "umi_qc": umi_summary,
            "program_cards": card_summary,
        },
        expected_files=[
            "outputs/figures/screen_analysis/NT466_B_vs_C_dotplot.pdf",
            "outputs/figures/screen_analysis/NT466_B_vs_C_dotplot.png",
            "outputs/figures/screen_analysis/NT466_B_vs_C_dotplot.svg",
        ],
    )


if __name__ == "__main__":
    main()
