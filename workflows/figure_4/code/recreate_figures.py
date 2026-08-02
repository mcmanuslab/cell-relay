#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "figure4_github_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
TABLES = WORKFLOW_ROOT / "data" / "figure_tables"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from package_utils import finish_run
import generate_publication_figures as source


def save(fig: plt.Figure, name: str) -> None:
    source.save_dual(fig, name)


def plot_umaps() -> None:
    umap = pd.read_csv(TABLES / "umap_standard_plot_points.csv")
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    order = ["dendritic cell", "C57BL/6 T cell", "OTI T cell"]
    colors = {
        "dendritic cell": "#ed8590",
        "OTI T cell": source.OTI_BLUE,
        "C57BL/6 T cell": source.C57_GREY,
    }
    plot_order = sorted(order, key=lambda label: int((umap["plot_label"] == label).sum()), reverse=True)
    handles = {}
    for label in plot_order:
        sub = umap[umap["plot_label"] == label]
        handles[label] = ax.scatter(
            sub["UMAP-1"], sub["UMAP-2"], s=1.15, alpha=0.82, linewidths=0,
            color=colors[label], label=label, rasterized=False,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend([handles[label] for label in order], order, markerscale=5.4, fontsize=5, loc="best")
    save(fig, "umap_standard_cell_types")

    background = pd.read_csv(TABLES / "umap_single_cell_interaction_background_plot_points.csv")
    links = pd.read_csv(TABLES / "umap_single_cell_interaction_links.csv")
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.scatter(
        background["UMAP-1"], background["UMAP-2"], s=0.85, color="#d9d9d9",
        alpha=0.24, linewidths=0, rasterized=False, zorder=0,
    )
    if not links.empty:
        umi_min = float(links["UMI"].min())
        umi_max = float(links["UMI"].max())
        for row in links.itertuples(index=False):
            fraction = (float(row.UMI) - umi_min) / max(umi_max - umi_min, 1.0)
            ax.plot(
                [row.donor_UMAP_1, row.recipient_UMAP_1],
                [row.donor_UMAP_2, row.recipient_UMAP_2],
                color="#ed8590", alpha=0.06 + 0.34 * fraction, linewidth=0.35, zorder=1,
            )
        donor = links.drop_duplicates("donor_CellBC")
        recipient = links.drop_duplicates("CellBC")
        ax.scatter(donor["donor_UMAP_1"], donor["donor_UMAP_2"], s=1.4, color="#ed8590", alpha=0.72, label="DCBC donor DC", linewidths=0, zorder=3)
        ax.scatter(recipient["recipient_UMAP_1"], recipient["recipient_UMAP_2"], s=1.1, color="#a68ff8", alpha=0.72, label="T cell recipient", linewidths=0, zorder=2)
        gradient_ax = ax.inset_axes([0.59, 0.075, 0.28, 0.035])
        rgba = np.array(matplotlib.colors.to_rgba("#ed8590"))
        gradient = np.ones((1, 128, 4), dtype=float)
        gradient[:, :, :3] = rgba[:3]
        gradient[:, :, 3] = np.linspace(0.06, 0.40, 128)
        gradient_ax.imshow(gradient, aspect="auto", extent=[umi_min, umi_max, 0, 1])
        gradient_ax.set_yticks([])
        gradient_ax.set_xticks([umi_min, umi_max])
        gradient_ax.set_xticklabels([f"{umi_min:.0f}", f"{umi_max:.0f}"], fontsize=4.8)
        gradient_ax.set_title("Link opacity (UMI)", fontsize=5, pad=1)
        for spine in gradient_ax.spines.values():
            spine.set_visible(False)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2, fontsize=5, loc="best")
    ax.set_title(f"DCBC transfer links, UMI >= {int(links['UMI'].min())}", fontsize=7)
    save(fig, "umap_single_cell_interaction_map")


def plot_pickup() -> None:
    multiplicity = pd.read_csv(TABLES / "unique_dcbc_pickup_multiplicity_source.csv")
    max_count = int(multiplicity["unique_dcbc"].max())
    fig, ax = plt.subplots(figsize=(max(4.0, 0.12 * (max_count + 1) + 1.4), 2.3))
    width = 0.42
    x = np.arange(max_count + 1)
    for offset, t_type, color in [(-width / 2, "OTI", source.OTI_BLUE), (width / 2, "C57BL6", source.C57_GREY)]:
        sub = multiplicity[multiplicity["t_cell_type"] == t_type].set_index("unique_dcbc")
        ax.bar(x + offset, [sub.loc[i, "fraction_t_cells"] for i in x], width=width, color=color, linewidth=0, label=t_type)
    ax.set_xlabel("Unique DCBCs picked up")
    ax.set_ylabel("Fraction of T cells")
    ax.set_xticks(x[:: max(1, math.ceil(len(x) / 18))])
    ax.legend(fontsize=6)
    save(fig, "unique_dcbc_pickup_multiplicity_bar")

    dots = pd.read_csv(TABLES / "unique_dcbc_pickup_single_cell_dot_source.csv")
    rng = np.random.default_rng(475)
    fig, ax = plt.subplots(figsize=(2.25, 2.7))
    cmap = LinearSegmentedColormap.from_list("umi_blue_red", ["#cfeaff", "#84c7ff", "#4e9bd7", "#ed8590", "#c83f58"])
    for x0, t_type in enumerate(["OTI", "C57BL6"]):
        sub = dots[dots["t_cell_type"] == t_type]
        jitter = rng.normal(0, 0.055, len(sub))
        ax.scatter(np.full(len(sub), x0) + jitter, sub["unique_dcbc"], c=sub["log10_total_dcbc_umi_plus1"], cmap=cmap, s=6, alpha=0.62, linewidths=0, rasterized=False)
        if len(sub):
            median = float(np.median(sub["unique_dcbc"].to_numpy(float)))
            ax.plot([x0 - 0.24, x0 + 0.24], [median, median], color="#ffffff", linewidth=2.0, solid_capstyle="round", zorder=5)
            ax.plot([x0 - 0.24, x0 + 0.24], [median, median], color="#222222", linewidth=1.0, solid_capstyle="round", zorder=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["OTI", "C57BL/6"])
    ax.set_ylabel("Unique DCBCs picked up")
    ax.set_xlim(-0.45, 1.45)
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=float(dots["log10_total_dcbc_umi_plus1"].min()), vmax=float(dots["log10_total_dcbc_umi_plus1"].max())))
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.06, pad=0.04)
    colorbar.set_label("log10(UMI + 1)", fontsize=6)
    save(fig, "unique_dcbc_pickup_single_cell_dot")


def plot_bubbles(peptides: list[str], treatments: list[str]) -> None:
    bubble = pd.read_csv(TABLES / "bubble_peptide_treatment_source.csv")
    source.plot_condition_bubbles(bubble, peptides, treatments, [])

    bubble = pd.read_csv(TABLES / "bubble_dc_supported_normalized_source.csv")
    color_col = "geomean_condition_umi_per_10k_dc_dcbc_umi"
    size_col = "qualifying_cells_per_dc_supported_unique_dcbc"
    positive = bubble.loc[bubble[color_col] > 0, color_col].to_numpy(float)
    vmin = float(np.percentile(positive, 5)) if len(positive) else 0.0
    vmax = float(np.percentile(positive, 90)) if len(positive) else 1.0
    vmax = vmax if vmax > vmin else vmin + 1e-9
    size_max = max(float(bubble[size_col].max()), 1e-12)
    size_values = bubble.loc[bubble[size_col] > 0, size_col].to_numpy(float)
    legend_values = np.unique(np.round(np.percentile(size_values, [25, 50, 90]), 2)) if len(size_values) else np.array([])
    cmap = LinearSegmentedColormap.from_list("dc_supported_transfer_red", ["#f7fbff", "#f9d6d6", "#ed8590", "#c83f58"])
    for t_type in source.T_CELL_TYPES:
        plot_df = bubble[(bubble["t_cell_type"] == t_type) & (bubble[color_col] > 0)].copy()
        fig, ax = plt.subplots(figsize=(2.9, 4.6))
        ax.set_axisbelow(True)
        if len(plot_df):
            sizes = 18 + (plot_df[size_col].to_numpy(float) / size_max) * 230
            colors = np.clip(plot_df[color_col].to_numpy(float), vmin, vmax)
            ax.scatter(plot_df["treatment_order"], plot_df["peptide_order"], s=sizes, c=colors, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="#7a2f36", linewidths=0.35, zorder=3)
        ax.set_xticks(range(len(treatments)))
        ax.set_xticklabels(treatments, rotation=35, ha="right")
        ax.set_yticks(range(len(peptides)))
        ax.set_yticklabels(peptides)
        ax.set_ylim(len(peptides) - 0.5, -0.5)
        ax.set_xlim(-0.5, len(treatments) - 0.5)
        ax.grid(True, color="#e8e8e8", linewidth=0.35, zorder=0)
        ax.set_title(f"{t_type} DC-supported")
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Peptide")
        scalar = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=ax, fraction=0.055, pad=0.04)
        colorbar.set_label("Geomean UMI / 10k DC UMI", fontsize=6)
        if len(legend_values):
            handles = [ax.scatter([], [], s=18 + (float(value) / size_max) * 230, facecolor="#f9d6d6", edgecolor="#7a2f36", linewidths=0.35) for value in legend_values]
            ax.legend(handles, [f"{value:g}" for value in legend_values], title="Cells / DC-supported DCBC", fontsize=5, title_fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=len(legend_values), columnspacing=1.0, handletextpad=0.7)
        save(fig, f"bubble_peptide_treatment_dc_supported_normalized_{t_type}")


def plot_treatment_boxplots(treatments: list[str]) -> None:
    data = pd.read_csv(TABLES / "tcell_treatment_dc_supported_normalized_peptide_collapsed_single_cell_source.csv")
    statistics = pd.read_csv(TABLES / "tcell_treatment_dc_supported_normalized_stats_vs_no_treatment.csv")
    metric = "treatment_umi_per_10k_dc_dcbc_umi"
    for t_type in source.T_CELL_TYPES:
        subset = data[data["t_cell_type"] == t_type]
        arrays = [subset.loc[subset["treatment"] == treatment, metric].to_numpy(float) for treatment in treatments]
        fig, ax = plt.subplots(figsize=(3.45, 2.6))
        boxplot = ax.boxplot(arrays, patch_artist=True, showfliers=False, widths=0.45)
        for patch, treatment in zip(boxplot["boxes"], treatments):
            patch.set(facecolor=source.TREATMENT_COLORS.get(treatment, "#cfcfcf"), edgecolor="#3f3f3f", alpha=0.85, linewidth=0.7)
        for element in ["whiskers", "caps", "medians"]:
            for artist in boxplot[element]:
                artist.set(color="#3f3f3f", linewidth=0.7)
        for index, values in enumerate(arrays, start=1):
            if len(values):
                ax.scatter(index, source.geometric_mean(values), marker="D", s=9, color="#7a2f36", edgecolor="#3f3f3f", linewidth=0.25, zorder=4)
            ax.text(index, -0.16, f"n={len(values):,}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(treatments) + 1))
        ax.set_xticklabels(["no_treat", "LPS", "PolyIC", "IFNg"])
        ax.set_ylabel("T-cell DCBC UMI / 10k DC UMI")
        ax.set_title(f"{t_type} peptide-collapsed, DC-supported")
        stat_subset = statistics[statistics["t_cell_type"] == t_type]
        y_top = ax.get_ylim()[1]
        for index, treatment in enumerate(["LPS", "PolyIC", "IFNg"], start=2):
            labels = stat_subset.loc[stat_subset["treatment"] == treatment, "q_label"]
            ax.text(index, y_top / 1.25, labels.iloc[0] if len(labels) else "q=NA", ha="center", va="center", fontsize=6)
        save(fig, f"tcell_treatment_dc_supported_normalized_peptide_collapsed_box_{t_type}")


def plot_peptide_lfc() -> None:
    table = pd.read_csv(TABLES / "peptide_umi_normalized_vs_bulk_lfc_source.csv")
    plot_df = table.dropna(subset=["bulk_log2_lfc_centered", "log2_lfc_t_over_dc_centered"]).copy()
    x = plot_df["bulk_log2_lfc_centered"].to_numpy(float)
    y = plot_df["log2_lfc_t_over_dc_centered"].to_numpy(float)
    regression = stats.linregress(x, y) if len(plot_df) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else None
    pearson = stats.pearsonr(x, y) if regression is not None else None
    spearman = stats.spearmanr(x, y) if regression is not None else None
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    if regression is not None:
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(line_x, regression.intercept + regression.slope * line_x, color="#c83f58", linewidth=0.9, zorder=1)
    colors = np.where(plot_df["is_control_center_peptide"], source.CONTROL_GREY, source.SKY_BLUE)
    ax.scatter(x, y, s=24, color=colors, edgecolors="none", linewidths=0, zorder=2)
    for row in plot_df.itertuples(index=False):
        ax.text(row.bulk_log2_lfc_centered, row.log2_lfc_t_over_dc_centered, row.PeptideBC_Name, fontsize=5, ha="left", va="bottom")
    ax.axhline(0, color="#777777", linewidth=0.5)
    ax.axvline(0, color="#777777", linewidth=0.5)
    if pearson is not None and spearman is not None:
        ax.text(0.03, 0.97, f"Pearson r={pearson.statistic:.2f}, p={pearson.pvalue:.2g}\nSpearman rho={spearman.statistic:.2f}, p={spearman.pvalue:.2g}", transform=ax.transAxes, ha="left", va="top", fontsize=5.5, bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.82})
    ax.set_xlabel("Bulk screen log2 fold change, centered")
    ax.set_ylabel("T/DC barcode proportion log2 fold change, centered")
    save(fig, "peptide_umi_normalized_vs_bulk_lfc")


def plot_dge() -> None:
    primary = pd.read_csv(TABLES / "dc_dge_top_gene_heatmap_primary_source.csv")
    columns = ["no_treatment", "LPS", "IFNg"]
    gene_order = primary[["gene", "zscore_cluster_order"]].drop_duplicates().sort_values("zscore_cluster_order")["gene"]
    pivot = primary.pivot(index="gene", columns="treatment_group", values="z").reindex(index=gene_order, columns=columns)
    fig, ax = plt.subplots(figsize=(2.1, max(2.9, 0.135 * len(pivot))))
    cmap = LinearSegmentedColormap.from_list("primary_dc_dge_heatmap", [source.HEATMAP_LOW, source.HEATMAP_MID, source.HEATMAP_HIGH])
    image = ax.imshow(np.clip(pivot.to_numpy(float), -2.5, 2.5), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=5.8)
    ax.set_title("Primary DC program genes", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02, label="z")
    save(fig, "dc_dge_top_gene_heatmap_primary")

    for treatment in ["LPS", "IFNg"]:
        table = pd.read_csv(TABLES / f"dc_dge_volcano_{treatment}_vs_no_treatment_source.csv")
        fig, ax = plt.subplots(figsize=(4.1, 3.1))
        other = table[~table["sig"]]
        significant = table[table["sig"] & ~table["in_top_gene_heatmap"]]
        highlighted = table[table["sig"] & table["in_top_gene_heatmap"]]
        ax.scatter(other["log2fc_mean_norm"], other["neglog10_padj"], s=2, color="#cfcfcf", alpha=0.45, linewidths=0)
        ax.scatter(significant["log2fc_mean_norm"], significant["neglog10_padj"], s=4, color="#ed8590", alpha=0.75, linewidths=0)
        ax.scatter(highlighted["log2fc_mean_norm"], highlighted["neglog10_padj"], s=10, color="#c83f58", alpha=0.95, edgecolors="#333333", linewidths=0.15)
        for _, row in highlighted.sort_values(["padj_bh", "log2fc_mean_norm"], ascending=[True, False]).head(15).iterrows():
            ax.text(row["log2fc_mean_norm"], row["neglog10_padj"], row["gene"], fontsize=5.5, color="#222222")
        ax.axvline(-0.5, color="#bbbbbb", linewidth=0.3)
        ax.axvline(0.5, color="#bbbbbb", linewidth=0.3)
        ax.axhline(-math.log10(0.05), color="#bbbbbb", linewidth=0.3)
        ax.set_xlabel("log2 fold-change of mean normalized expression")
        ax.set_ylabel("-log10 BH-adjusted p")
        ax.set_title(f"DC DGE: {treatment} vs no_treatment")
        save(fig, f"dc_dge_volcano_{treatment}_vs_no_treatment")

    programs = list(reversed(list(source.DC_PROGRAMS)))
    program_table = pd.read_csv(TABLES / "dc_dge_program_summary.csv")
    fig, ax = plt.subplots(figsize=(3.51, 3.0))
    y = np.arange(len(programs))
    width = 0.34
    for offset, treatment, color in [(-width / 2, "LPS", "#f2aa59"), (width / 2, "IFNg", "#6dead4")]:
        subset = program_table[program_table["treatment"] == treatment].set_index("program").reindex(programs)
        ax.barh(y + offset, subset["median_log2fc"], height=width, color=color, edgecolor="white", linewidth=0.2, label=treatment)
    ax.axvline(0, color="#777777", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(programs)
    ax.set_xlabel("Median program log2FC vs no_treatment")
    ax.set_title("DC gene-program shifts")
    ax.legend(fontsize=6)
    save(fig, "dc_dge_program_summary")

    heatmap = pd.read_csv(TABLES / "dc_dge_top_gene_heatmap_source.csv")
    heatmap_genes = list(dict.fromkeys(heatmap["gene"].tolist()))
    pivot = heatmap.pivot(index="gene", columns="treatment_group", values="z").reindex(index=list(reversed(heatmap_genes)), columns=columns)
    fig, ax = plt.subplots(figsize=(2.4, max(3.2, 0.095 * len(pivot))))
    cmap = LinearSegmentedColormap.from_list("dc_dge_heatmap", [source.HEATMAP_LOW, source.HEATMAP_MID, source.HEATMAP_HIGH])
    image = ax.imshow(np.clip(pivot.to_numpy(float), -2.5, 2.5), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=5.5)
    ax.set_title("Top treatment-up DC genes")
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02, label="z")
    save(fig, "dc_dge_top_gene_heatmap")


def plot_clustered_heatmaps(peptides: list[str], treatments: list[str]) -> None:
    specifications = [
        ("dc_phenotype_heatmap_zscore.csv", treatments, "DC phenotype heatmap", "dc_phenotype_heatmap_clustered"),
        ("t_cell_treatment_heatmap_zscore.csv", ["no_treatment", "LPS", "PolyIC", "IFNg", "Multiple", "OTI no interaction", "C57BL/6"], "T cell phenotype heatmap by treatment", "t_cell_treatment_heatmap_clustered"),
        ("t_cell_peptide_heatmap_zscore.csv", peptides + ["multi_peptide", "no_interaction"], "OTI T cell phenotype heatmap by peptide", "t_cell_peptide_heatmap_clustered"),
    ]
    for filename, columns, title, name in specifications:
        table_path = TABLES / filename
        source.clustered_heatmap(pd.read_csv(table_path), columns, title, name, [], table_path)


def plot_gene_set_bubble() -> None:
    table = pd.read_csv(TABLES / "t_cell_select_gene_set_bubble_source.csv")
    plot_df = table.dropna(subset=["zscore"]).copy()
    columns = table[["group", "x"]].drop_duplicates().sort_values("x")["group"].tolist()
    size_max = max(float(plot_df["positive_mean_expression"].max()), 1e-12)
    size_for = lambda value: 12 + (np.asarray(value, dtype=float) / size_max) * 145
    cmap = LinearSegmentedColormap.from_list("zdiv", ["#5d8af7", "#ffffff", "#ed8590"])
    fig, ax = plt.subplots(figsize=(4.45, 6.8))
    ax.scatter(plot_df["x_plot"], plot_df["y_plot"], s=size_for(plot_df["positive_mean_expression"]), c=plot_df["zscore"], cmap=cmap, vmin=-2, vmax=2, edgecolors="#555555", linewidths=0.25)
    x_spacing = float(table.loc[table["x"] == 1, "x_plot"].iloc[0]) if (table["x"] == 1).any() else 1.12
    ax.set_xticks([i * x_spacing for i in range(len(columns))])
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(sorted(table["y_plot"].unique()))
    ax.set_yticklabels([])
    for row in table.drop_duplicates("y")[["gene", "y_plot"]].itertuples(index=False):
        ax.text(-0.72, row.y_plot, row.gene, fontsize=5.5, ha="right", va="center")
    max_x = (len(columns) - 1) * x_spacing
    for group, subset in table.groupby("gene_set", sort=False):
        start, end = int(subset["y"].min()), int(subset["y"].max())
        y_spacing = float(subset["y_plot"].max() / end) if end else 1.46
        ax.text(max_x + 0.55, ((start + end) / 2) * y_spacing, group, fontsize=6, va="center", ha="left")
        ax.axhline((end + 0.5) * y_spacing, color="#dddddd", linewidth=0.5)
    ax.set_xlim(-0.86, max_x + 1.95)
    ax.set_ylim(table["y_plot"].max() + 0.9, -0.9)
    ax.set_title("Select T cell phenotype genes")
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-2, vmax=2))
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("Z-score", fontsize=6)
    positive = plot_df.loc[plot_df["positive_mean_expression"] > 0, "positive_mean_expression"].to_numpy(float)
    if len(positive):
        values = np.unique(np.round(np.percentile(positive, [25, 50, 90]), 2))
        handles = [ax.scatter([], [], s=float(size_for(value)), facecolor="#cfcfcf", edgecolor="#555555", linewidth=0.25) for value in values]
        ax.legend(handles, [f"{value:g}" for value in values], title="Mean expr+ cells", fontsize=5, title_fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(values), columnspacing=1.2, handletextpad=0.8)
    save(fig, "t_cell_select_gene_set_bubble")


def plot_signature_figures() -> None:
    heatmap = pd.read_csv(TABLES / "t_cell_signature_group_median_heatmap_zscore.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    cmap = LinearSegmentedColormap.from_list("signature_heatmap", [source.HEATMAP_LOW, source.HEATMAP_MID, source.HEATMAP_HIGH])
    image = ax.imshow(np.clip(heatmap.to_numpy(float), -2.5, 2.5), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(["multi" if group == "multi_peptide" else group for group in heatmap.columns], rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index, fontsize=6)
    ax.set_title("OTI peptide-group T-cell signature medians", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label="Row z-score")
    save(fig, "t_cell_signature_group_median_heatmap")

    effects = pd.read_csv(TABLES / "tcell_signature_effect_ci_vs_no_interaction_source.csv")
    group_order = effects[["peptide_group", "peptide_order"]].drop_duplicates().sort_values("peptide_order")["peptide_group"].tolist()
    signature_order = effects[["signature", "signature_order"]].drop_duplicates().sort_values("signature_order")["signature"].tolist()
    color_col = "geomean_dominant_peptide_umi_per_10k_dc_dcbc_umi"
    color_max = max(float(effects[color_col].replace([np.inf, -np.inf], np.nan).max()), 1.0)
    cmap = LinearSegmentedColormap.from_list("dominant_peptide_transfer_red", ["#f7fbff", "#f9d6d6", "#ed8590", "#c83f58"])
    norm = Normalize(vmin=0.0, vmax=color_max)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for ax, signature in zip(axes.ravel(), signature_order):
        subset = effects[effects["signature"] == signature].sort_values("peptide_order")
        x = subset["peptide_order"].to_numpy(float)
        y = subset["median_diff"].to_numpy(float)
        yerr = np.vstack([y - subset["median_diff_ci95_low"].to_numpy(float), subset["median_diff_ci95_high"].to_numpy(float) - y])
        metric = subset[color_col].replace([np.inf, -np.inf], np.nan)
        has_metric = metric.notna().to_numpy()
        ax.axhline(0, color="#9a9a9a", linewidth=0.45)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#555555", elinewidth=0.45, capsize=1.8, capthick=0.45)
        ax.scatter(x[~has_metric], y[~has_metric], s=22, color="#d7bde2", edgecolors="#333333", linewidths=0.25, zorder=3)
        ax.scatter(x[has_metric], y[has_metric], s=22, c=metric.loc[has_metric].to_numpy(float), cmap=cmap, norm=norm, edgecolors="#333333", linewidths=0.25, zorder=3)
        significant = subset["q_value_bh_by_signature"].to_numpy(float) < 0.05
        ax.scatter(x[significant], y[significant], s=44, facecolors="none", edgecolors="#333333", linewidths=0.45, zorder=4)
        ax.set_title(signature, fontsize=8)
        ax.set_ylabel("Median score difference")
        ax.grid(axis="y", color="#eeeeee", linewidth=0.35)
    for ax in axes[-1, :]:
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(["multi" if group == "multi_peptide" else group for group in group_order], rotation=45, ha="right", fontsize=5.5)
    fig.suptitle("Program score shifts vs OTI no-interaction", fontsize=9)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), fraction=0.025, pad=0.018)
    colorbar.set_label("Geomean peptide UMI per 10k DC UMI", fontsize=6)
    colorbar.ax.tick_params(labelsize=5)
    save(fig, "tcell_signature_effect_ci_vs_no_interaction")

    bubble = pd.read_csv(TABLES / "tcell_highlighted_gene_relative_pattern_bubble_source.csv")
    group_order = bubble[["peptide_group", "peptide_order"]].drop_duplicates().sort_values("peptide_order")["peptide_group"].tolist()
    genes = bubble[["gene", "gene_order"]].drop_duplicates().sort_values("gene_order")["gene"].tolist()
    cmap = LinearSegmentedColormap.from_list("relative_gene_bubble", [source.HEATMAP_LOW, source.HEATMAP_MID, source.HEATMAP_HIGH])
    norm = TwoSlopeNorm(vcenter=0, vmin=-2.2, vmax=2.2)
    fig, ax = plt.subplots(figsize=(6.9, max(4.4, 0.16 * len(genes))))
    nonsignificant = ~bubble["is_sig_abs0p5_fdr0p05"].fillna(False)
    ax.scatter(bubble.loc[nonsignificant, "x"], bubble.loc[nonsignificant, "y"], s=bubble.loc[nonsignificant, "plot_size"], c=bubble.loc[nonsignificant, "gene_centered_log2fc_z"], cmap=cmap, norm=norm, edgecolors="#c0c0c0", linewidths=0.08, alpha=0.9)
    scatter = ax.scatter(bubble.loc[~nonsignificant, "x"], bubble.loc[~nonsignificant, "y"], s=bubble.loc[~nonsignificant, "plot_size"], c=bubble.loc[~nonsignificant, "gene_centered_log2fc_z"], cmap=cmap, norm=norm, edgecolors="#333333", linewidths=0.26, alpha=0.96)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(["multi" if group == "multi_peptide" else group for group in group_order], rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(list(reversed(genes)), fontsize=5.5)
    ax.set_xlabel("Peptide group")
    ax.set_ylabel("Highlighted gene")
    ax.set_title("Relative peptide pattern of highlighted genes", fontsize=8)
    ax.set_xlim(-0.55, len(group_order) - 0.45)
    ax.set_ylim(-0.7, len(genes) - 0.3)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label("Gene-centered log2FC z-score", fontsize=6)
    for percent in [25, 50, 75]:
        ax.scatter([], [], s=5 + (percent / 100) * 72, color="#eeeeee", edgecolors="#555555", linewidths=0.15, label=f"{percent}%")
    ax.legend(title="Detected", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=5.3, title_fontsize=5.5, borderaxespad=0)
    save(fig, "tcell_highlighted_gene_relative_pattern_bubble")


def plot_umi_and_interactions(peptides: list[str]) -> None:
    dc = pd.read_csv(TABLES / "dc_umi_per_cell_violin_source.csv")
    treatments = ["no_treatment", "LPS", "PolyIC", "IFNg"]
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    arrays = [np.log10(dc.loc[dc["AssignedTreatment"] == treatment, "total_barcode_umi"].astype(float) + 1) for treatment in treatments]
    source.violin(ax, arrays, list(range(len(treatments))), [source.C57_GREY for _ in treatments])
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(treatments, rotation=35, ha="right")
    ax.set_ylabel("log10(DCBC UMI + 1)")
    ax.set_title("DC UMI per cell")
    save(fig, "dc_umi_per_cell_by_treatment_violin")

    t_cells = pd.read_csv(TABLES / "t_cell_umi_per_cell_violin_source.csv")
    fig, ax = plt.subplots(figsize=(1.9, 2.1))
    arrays = [np.log10(t_cells.loc[t_cells["t_cell_type"] == group, "total_barcode_umi"].astype(float) + 1) for group in ["OTI", "C57BL6"]]
    source.violin(ax, arrays, [0, 1], [source.OTI_BLUE, source.C57_GREY], widths=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["OTI", "C57BL/6"])
    ax.set_ylabel("log10(DCBC UMI + 1)")
    ax.set_title("T cell UMI per cell")
    save(fig, "t_cell_umi_per_cell_violin")

    fingerprints = pd.read_csv(TABLES / "single_cell_interaction_fingerprints_source.csv")
    metrics = pd.read_csv(TABLES / "interaction_strength_distribution_source.csv")
    cell_meta = fingerprints[["CellBC", "t_cell_type", "fingerprint_level"]].drop_duplicates().merge(metrics, on=["CellBC", "t_cell_type"], how="left")
    for t_type in ["OTI", "C57BL6"]:
        cells = cell_meta[cell_meta["t_cell_type"] == t_type].copy()
        cells["level_order"] = cells["fingerprint_level"].map({"high": 0, "low": 1})
        cells = cells.sort_values(["level_order", "unique_dcbc", "total_dcbc_umi"], ascending=[True, False, False])
        plot_df = fingerprints[fingerprints["CellBC"].isin(cells["CellBC"])]
        cell_order = cells["CellBC"].tolist()
        fig, ax = plt.subplots(figsize=(4.7, max(3.0, 0.15 * len(cell_order) + 1.15)))
        left = np.zeros(len(cell_order))
        y = np.arange(len(cell_order))
        for peptide in peptides:
            values = np.array([plot_df.loc[(plot_df["CellBC"] == cell) & (plot_df["PeptideBC_Name"] == peptide), "proportion"].sum() for cell in cell_order])
            ax.barh(y, values, left=left, color=source.PEPTIDE_COLORS[peptide], linewidth=0, label=peptide)
            left += values
        counts = {"high": 0, "low": 0}
        labels = []
        for level in cells["fingerprint_level"]:
            counts[level] += 1
            labels.append(f"{level[0].upper()}{counts[level]}")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        high_n = int((cells["fingerprint_level"] == "high").sum())
        if 0 < high_n < len(cells):
            ax.axhline(high_n - 0.5, color="#555555", linewidth=0.55)
            ax.text(1.01, (high_n - 1) / 2, "High", transform=ax.get_yaxis_transform(), fontsize=6, va="center", ha="left")
            ax.text(1.01, high_n + (len(cells) - high_n - 1) / 2, "Low", transform=ax.get_yaxis_transform(), fontsize=6, va="center", ha="left")
        ax.invert_yaxis()
        ax.set_xlabel("Peptide fraction of DCBC UMI")
        ax.set_ylabel("Sampled cell")
        ax.set_title(f"{t_type} interaction fingerprints")
        ax.legend(ncol=4, fontsize=4.8, loc="lower center", bbox_to_anchor=(0.5, -0.48))
        save(fig, f"interaction_fingerprint_{t_type}")

    curve = pd.read_csv(TABLES / "interaction_strength_distribution_curve_source.csv")
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    for t_type, color, label in [("OTI", source.OTI_BLUE, "OTI"), ("C57BL6", source.C57_GREY, "C57BL/6")]:
        subset = curve[curve["t_cell_type"] == t_type].sort_values("bin_center")
        ax.plot(subset["bin_center"], subset["n_t_cells"], color=color, linewidth=1.3, drawstyle="steps-mid", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Total DCBC UMI per T cell")
    ax.set_ylabel("Number of T cells")
    ax.set_title("Interaction strength")
    ax.legend(fontsize=6)
    save(fig, "interaction_strength_distribution")


def main() -> None:
    started = time.time()
    (WORKFLOW_ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    (WORKFLOW_ROOT / "outputs" / "summaries").mkdir(parents=True, exist_ok=True)
    config = json.loads((WORKFLOW_ROOT / "code" / "config.yaml").read_text())
    peptides = config["orders"]["peptides"]
    treatments = config["orders"]["treatments"]

    plot_umaps()
    plot_pickup()
    plot_bubbles(peptides, treatments)
    plot_treatment_boxplots(treatments)
    plot_peptide_lfc()
    plot_dge()
    plot_clustered_heatmaps(peptides, treatments)
    plot_gene_set_bubble()
    plot_signature_figures()
    plot_umi_and_interactions(peptides)

    manifest = pd.read_csv(TABLES / "figure_manifest.csv")
    expected = []
    for column in ["pdf", "png"]:
        expected.extend(f"outputs/figures/{Path(value).name}" for value in manifest[column])
    finish_run(
        WORKFLOW_ROOT,
        "Figure 4",
        started,
        [TABLES, WORKFLOW_ROOT / "code" / "config.yaml"],
        details={"manifest_rows": len(manifest), "recreated_figures": len(list((WORKFLOW_ROOT / "outputs" / "figures").glob("*.pdf")))},
        expected_files=expected,
    )


if __name__ == "__main__":
    main()
