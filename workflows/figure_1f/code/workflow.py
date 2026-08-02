#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
FIGURE_TABLE_DIR = ROOT / "data" / "figure_tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
SUMMARY_DIR = ROOT / "outputs" / "summaries"

for directory in [INTERMEDIATE_DIR, FIGURE_TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

matplotlib_cache = Path(tempfile.gettempdir()) / "figure_1f_matplotlib_cache"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

PMHC_PATTERNS = [
    ("B08_ELRRKMMYM", "aaggatgaacacgacnnnnn"),
    ("A02_NLVPMVATV", "aaggatgaacnnnnnaccgg"),
    ("A02_ELAGIGILTV", "aaggannnnnacgacaccgg"),
    ("A02_SLLMWITQV", "nnnnntgaacacgacaccgg"),
]

PMHC_ORDER = [
    "B08_ELRRKMMYM",
    "A02_NLVPMVATV",
    "A02_ELAGIGILTV",
    "A02_SLLMWITQV",
]

PEPTIDE_LABELS = {
    "B08_ELRRKMMYM": "P1",
    "A02_NLVPMVATV": "P2",
    "A02_ELAGIGILTV": "P3",
    "A02_SLLMWITQV": "P4",
}

SAMPLE_ORDER = ["J1", "J2", "J3", "J4"]
K_SAMPLE_ORDER = ["K1", "K2", "K3", "K4"]
BASES = ["a", "c", "g", "t"]
RC_TRANS = str.maketrans("ACGTN", "TGCAN")
PSEUDOCOUNT = 0.5


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(clean_for_json(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_raw_manifest() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".gz":
            role = "sequencing_fastq_used"
        elif path.suffix.lower() == ".fcs":
            role = "fcs_copied_for_raw_context_not_used"
        else:
            role = "raw_or_provenance_file"
        entries.append(
            {
                "relative_path": relpath(path),
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest_path = SUMMARY_DIR / "raw_data_manifest.csv"
    pd.DataFrame(entries).to_csv(manifest_path, index=False)
    return entries


def expand_barcode_pattern(pattern: str) -> list[str]:
    n_positions = [i for i, char in enumerate(pattern) if char == "n"]
    expanded: list[str] = []
    for nts in product(BASES, repeat=len(n_positions)):
        seq = list(pattern)
        for pos, nt in zip(n_positions, nts):
            seq[pos] = nt
        expanded.append("".join(seq))
    return expanded


def generate_barcodes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for pmhc, pattern in PMHC_PATTERNS:
        for barcode in expand_barcode_pattern(pattern):
            rows.append({"pMHC": pmhc, "Barcode": barcode})

    generated_df = pd.DataFrame(rows, columns=["pMHC", "Barcode"])
    generated_df.to_csv(INTERMEDIATE_DIR / "pmhc_barcodes_generated.csv", index=False)

    barcode_df = generated_df.copy()
    barcode_df["pMHC"] = barcode_df["pMHC"].str.strip()
    barcode_df["Barcode"] = barcode_df["Barcode"].str.strip()
    barcode_df["Barcode_upper"] = barcode_df["Barcode"].str.upper()
    barcode_df = barcode_df[
        (barcode_df["pMHC"] != "") & (barcode_df["Barcode_upper"] != "")
    ].copy()
    barcode_df = barcode_df.drop_duplicates(subset=["pMHC", "Barcode_upper"]).reset_index(
        drop=True
    )

    pmhc_per_barcode = barcode_df.groupby("Barcode_upper")["pMHC"].nunique()
    colliding_barcodes = set(pmhc_per_barcode[pmhc_per_barcode > 1].index)

    removed_df = (
        barcode_df[barcode_df["Barcode_upper"].isin(colliding_barcodes)][
            ["pMHC", "Barcode"]
        ]
        .sort_values(["Barcode", "pMHC"])
        .reset_index(drop=True)
    )
    removed_df.to_csv(INTERMEDIATE_DIR / "pmhc_barcodes_collisions_removed.csv", index=False)

    filtered_df = (
        barcode_df[~barcode_df["Barcode_upper"].isin(colliding_barcodes)]
        .sort_values(["pMHC", "Barcode"])
        .reset_index(drop=True)
    )
    if filtered_df.empty:
        raise ValueError("No unique barcodes remain after removing cross-pMHC collisions.")

    filtered_df[["pMHC", "Barcode"]].to_csv(INTERMEDIATE_DIR / "pmhc_barcodes.csv", index=False)

    barcode_lengths = sorted(filtered_df["Barcode_upper"].str.len().unique())
    if len(barcode_lengths) != 1:
        raise ValueError(f"Expected one barcode length, found: {barcode_lengths}")

    qc = {
        "generated_rows": int(len(generated_df)),
        "filtered_rows": int(len(filtered_df)),
        "removed_collision_rows": int(len(removed_df)),
        "removed_collision_barcodes": int(len(colliding_barcodes)),
        "barcode_length": int(barcode_lengths[0]),
        "filtered_barcodes_per_pMHC": filtered_df["pMHC"].value_counts().sort_index().to_dict(),
    }
    return generated_df, filtered_df, removed_df, qc


def revcomp(seq: str) -> str:
    return seq.upper().translate(RC_TRANS)[::-1]


def sample_name_from_fastq(path: Path, seen: Counter[str]) -> str:
    name = path.name
    if name.endswith(".fastq.gz"):
        base = name[:-9]
    elif name.endswith(".fq.gz"):
        base = name[:-6]
    else:
        base = path.stem

    prefix = base.split("_")[0]
    seen[prefix] += 1
    if seen[prefix] == 1:
        return prefix
    return f"{prefix}__{seen[prefix]}"


def count_barcodes_in_fastq(
    fastq_path: Path, barcode_lookup: dict[str, str], barcode_length: int
) -> tuple[Counter[str], dict[str, Any]]:
    counts: Counter[str] = Counter()
    read_count = 0
    reads_with_match = 0
    total_matches = 0

    with gzip.open(fastq_path, "rt") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            seq_line = handle.readline()
            plus = handle.readline()
            qual = handle.readline()

            if not seq_line or not plus or not qual:
                raise ValueError(f"Malformed FASTQ record in {fastq_path}")

            seq = seq_line.strip().upper()
            read_count += 1
            if len(seq) < barcode_length:
                continue

            matches_in_read = 0
            for start in range(len(seq) - barcode_length + 1):
                matched_barcode = barcode_lookup.get(seq[start : start + barcode_length])
                if matched_barcode is None:
                    continue
                counts[matched_barcode] += 1
                matches_in_read += 1

            if matches_in_read:
                reads_with_match += 1
                total_matches += matches_in_read

    qc = {
        "fastq": relpath(fastq_path),
        "reads_scanned": read_count,
        "reads_with_at_least_one_match": reads_with_match,
        "total_barcode_matches": total_matches,
        "unique_barcodes_observed": len(counts),
    }
    return counts, qc


def build_count_table(filtered_barcodes: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    barcode_lengths = filtered_barcodes["Barcode_upper"].str.len().unique()
    if len(barcode_lengths) != 1:
        raise ValueError(f"Expected one barcode length, found: {sorted(barcode_lengths)}")
    barcode_length = int(barcode_lengths[0])

    barcode_df = filtered_barcodes.copy()
    barcode_df["Barcode_RC"] = barcode_df["Barcode_upper"].map(revcomp)
    barcode_lookup = dict(zip(barcode_df["Barcode_RC"], barcode_df["Barcode_upper"]))

    fastq_files = sorted((RAW_DIR / "NT222-D03").glob("*.fastq.gz"), key=lambda path: path.name)
    if not fastq_files:
        raise FileNotFoundError(f"No FASTQ files found in {(RAW_DIR / 'NT222-D03').resolve()}")

    result_df = barcode_df[["pMHC", "Barcode", "Barcode_upper"]].copy()
    sample_columns: list[str] = []
    sample_qc: list[dict[str, Any]] = []
    seen_sample_names: Counter[str] = Counter()

    for fastq_path in fastq_files:
        sample_name = sample_name_from_fastq(fastq_path, seen_sample_names)
        counts, qc = count_barcodes_in_fastq(fastq_path, barcode_lookup, barcode_length)
        result_df[sample_name] = result_df["Barcode_upper"].map(counts).fillna(0).astype(int)
        sample_columns.append(sample_name)
        qc["sample"] = sample_name
        sample_qc.append(qc)

    count_df = result_df[["pMHC", "Barcode"] + sample_columns]
    count_df.to_csv(INTERMEDIATE_DIR / "pmhc_barcode_counts.csv", index=False)
    return count_df, sample_qc


def canonical_sample_name(col_name: str) -> str:
    return re.sub(r"__\d+$", "", str(col_name))


def compute_log2fc(
    counts_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata_cols = ["pMHC", "Barcode"]
    sample_cols = [col for col in counts_df.columns if col not in metadata_cols]

    sample_groups: dict[str, list[str]] = {}
    for col in sample_cols:
        sample_groups.setdefault(canonical_sample_name(col), []).append(col)

    df = counts_df[metadata_cols].copy()
    for canon_name, cols in sample_groups.items():
        df[canon_name] = (
            counts_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(float)
        )

    required_cols = K_SAMPLE_ORDER + SAMPLE_ORDER
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        available = [col for col in df.columns if col not in metadata_cols]
        raise ValueError(f"Missing required sample columns: {missing}; available: {available}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    df = df.sort_values(["pMHC", "Barcode"]).reset_index(drop=True)

    k_totals = df[K_SAMPLE_ORDER].sum(axis=0)
    zero_k = list(k_totals[k_totals <= 0].index)
    if zero_k:
        raise ValueError(f"K replicate(s) with zero total counts: {zero_k}")

    target_k_total_per_replicate = k_totals.mean()
    k_scale_factors = target_k_total_per_replicate / k_totals
    k_equalized_cols: list[str] = []
    for col in K_SAMPLE_ORDER:
        eq_col = f"{col}_equalized"
        df[eq_col] = df[col] * k_scale_factors[col]
        k_equalized_cols.append(eq_col)

    df["K"] = df[k_equalized_cols].sum(axis=1)
    k_total = df["K"].sum()

    k_scaling_df = pd.DataFrame(
        {
            "sample": K_SAMPLE_ORDER,
            "raw_total": [k_totals[col] for col in K_SAMPLE_ORDER],
            "scale_factor": [k_scale_factors[col] for col in K_SAMPLE_ORDER],
            "equalized_total": [df[f"{col}_equalized"].sum() for col in K_SAMPLE_ORDER],
        }
    )
    k_scaling_df["equalized_share_of_final_K"] = (
        k_scaling_df["equalized_total"] / k_scaling_df["equalized_total"].sum()
    )
    k_scaling_df.to_csv(FIGURE_TABLE_DIR / "K_equalization_summary.csv", index=False)

    j_totals = df[SAMPLE_ORDER].sum(axis=0)
    zero_j = list(j_totals[j_totals <= 0].index)
    if zero_j:
        raise ValueError(f"J sample(s) with zero total counts: {zero_j}")

    df["K_fraction"] = df["K"] / k_total
    for col in SAMPLE_ORDER:
        norm_col = f"{col}_normalized"
        frac_col = f"{col}_fraction"
        lfc_col = f"{col}_log2FC_vs_K"
        df[norm_col] = df[col] * (k_total / j_totals[col])
        df[frac_col] = df[norm_col] / k_total
        df[lfc_col] = np.log2((df[norm_col] + PSEUDOCOUNT) / (df["K"] + PSEUDOCOUNT))

    summary_frames = []
    for col in SAMPLE_ORDER:
        norm_col = f"{col}_normalized"
        lfc_col = f"{col}_log2FC_vs_K"

        grp = (
            df.groupby("pMHC", sort=True)
            .agg(
                n_barcodes=("Barcode", "size"),
                detected_barcodes=(col, lambda s: int((s > 0).sum())),
                sample_total_counts_raw=(col, "sum"),
                sample_total_counts_normalized=(norm_col, "sum"),
                K_total_counts=("K", "sum"),
                barcode_mean_log2FC=(lfc_col, "mean"),
                barcode_median_log2FC=(lfc_col, "median"),
                barcode_sd_log2FC=(lfc_col, "std"),
                barcode_min_log2FC=(lfc_col, "min"),
                barcode_q25_log2FC=(lfc_col, lambda s: s.quantile(0.25)),
                barcode_q75_log2FC=(lfc_col, lambda s: s.quantile(0.75)),
                barcode_max_log2FC=(lfc_col, "max"),
            )
            .reset_index()
        )
        grp["detected_fraction"] = grp["detected_barcodes"] / grp["n_barcodes"]
        grp["sample_total_fraction"] = grp["sample_total_counts_normalized"] / k_total
        grp["K_total_fraction"] = grp["K_total_counts"] / k_total
        grp["pMHC_log2FC_vs_K"] = np.log2(
            (grp["sample_total_counts_normalized"] + PSEUDOCOUNT)
            / (grp["K_total_counts"] + PSEUDOCOUNT)
        )
        grp.insert(0, "Sample", col)
        summary_frames.append(grp)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df = summary_df[
        [
            "Sample",
            "pMHC",
            "n_barcodes",
            "detected_barcodes",
            "detected_fraction",
            "sample_total_counts_raw",
            "sample_total_counts_normalized",
            "K_total_counts",
            "sample_total_fraction",
            "K_total_fraction",
            "pMHC_log2FC_vs_K",
            "barcode_mean_log2FC",
            "barcode_median_log2FC",
            "barcode_sd_log2FC",
            "barcode_min_log2FC",
            "barcode_q25_log2FC",
            "barcode_q75_log2FC",
            "barcode_max_log2FC",
        ]
    ].sort_values(["Sample", "pMHC"]).reset_index(drop=True)

    barcode_output_cols = (
        ["pMHC", "Barcode"]
        + K_SAMPLE_ORDER
        + k_equalized_cols
        + ["K", "K_fraction"]
        + SAMPLE_ORDER
    )
    for col in SAMPLE_ORDER:
        barcode_output_cols.extend([f"{col}_normalized", f"{col}_fraction", f"{col}_log2FC_vs_K"])

    barcode_level_df = df[barcode_output_cols].copy()
    barcode_level_df.to_csv(FIGURE_TABLE_DIR / "pmhc_barcode_J_vs_K_log2FC.csv", index=False)
    summary_df.to_csv(FIGURE_TABLE_DIR / "pmhc_pMHC_summary_J_vs_K.csv", index=False)

    qc = {
        "pseudocount": PSEUDOCOUNT,
        "raw_sample_totals": {col: float(df[col].sum()) for col in required_cols},
        "combined_equalized_K_total": float(k_total),
        "K_equalized_share_range": [
            float(k_scaling_df["equalized_share_of_final_K"].min()),
            float(k_scaling_df["equalized_share_of_final_K"].max()),
        ],
        "summary_rows": int(len(summary_df)),
        "barcode_level_rows": int(len(barcode_level_df)),
    }
    return barcode_level_df, summary_df, k_scaling_df, qc


def build_dotplot_tables(barcode_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pmhc_order = [p for p in PMHC_ORDER if p in barcode_df["pMHC"].unique()]
    pmhc_order += [p for p in barcode_df["pMHC"].unique() if p not in pmhc_order]

    x_cursor = 0
    ordered_chunks = []
    group_rows = []
    group_gap = 25

    for pmhc in pmhc_order:
        sub = barcode_df.loc[barcode_df["pMHC"] == pmhc].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("Barcode", ascending=True, kind="mergesort")
        n_rows = len(sub)
        sub["x_base"] = np.arange(x_cursor, x_cursor + n_rows, dtype=float)
        ordered_chunks.append(sub)
        group_rows.append(
            {
                "pMHC": pmhc,
                "start": x_cursor - 0.5,
                "end": x_cursor + n_rows - 0.5,
                "center": x_cursor + (n_rows - 1) / 2,
                "n_barcodes": n_rows,
            }
        )
        x_cursor += n_rows + group_gap

    ordered_df = pd.concat(ordered_chunks, ignore_index=True)
    sample_order = [s for s in SAMPLE_ORDER if f"{s}_log2FC_vs_K" in ordered_df.columns]
    if not sample_order:
        raise ValueError("No J*_log2FC_vs_K columns found for dot plot.")

    offsets = dict(zip(sample_order, np.linspace(-0.30, 0.30, len(sample_order))))
    rows = []
    for sample in sample_order:
        lfc_col = f"{sample}_log2FC_vs_K"
        tmp = ordered_df[["pMHC", "Barcode", "x_base", "K_fraction"]].copy()
        tmp["Sample"] = sample
        tmp["x"] = tmp["x_base"] + offsets[sample]
        tmp["Log2FC"] = ordered_df[lfc_col].to_numpy(dtype=float)
        rows.append(tmp)

    dot_df = pd.concat(rows, ignore_index=True)
    group_df = pd.DataFrame(group_rows)
    dot_df.to_csv(FIGURE_TABLE_DIR / "barcode_log2FC_dotplot_J_vs_K_table.csv", index=False)
    group_df.to_csv(FIGURE_TABLE_DIR / "barcode_log2FC_dotplot_J_vs_K_groups.csv", index=False)
    return dot_df, group_df


def plot_dotplot(dot_df: pd.DataFrame, group_df: pd.DataFrame) -> dict[str, str]:
    sample_order = [s for s in SAMPLE_ORDER if s in set(dot_df["Sample"])]
    all_y = pd.to_numeric(dot_df["Log2FC"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    all_y = all_y.dropna().to_numpy()
    if len(all_y) == 0:
        raise ValueError("No finite log2FC values found for dot plot.")

    abs_max = np.nanmax(np.abs(all_y))
    y_pad = max(0.35, abs_max * 0.06)
    y_lim = 5 if abs_max + y_pad > 5 else abs_max + y_pad

    n_barcodes = dot_df[["pMHC", "Barcode"]].drop_duplicates().shape[0]
    fig_width = float(np.clip(10 + n_barcodes / 650, 12, 18))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2), dpi=300)

    for idx, info in group_df.reset_index(drop=True).iterrows():
        if idx % 2 == 0:
            ax.axvspan(info["start"], info["end"], facecolor="0.97", edgecolor="none", zorder=0)

    for (_, left), (_, right) in zip(group_df.iloc[:-1].iterrows(), group_df.iloc[1:].iterrows()):
        separator_x = (left["end"] + right["start"]) / 2
        ax.axvline(separator_x, color="0.85", linewidth=0.8, zorder=1)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sample_colors = {sample: default_colors[i % len(default_colors)] for i, sample in enumerate(sample_order)}

    for sample in sample_order:
        sub = dot_df.loc[dot_df["Sample"] == sample]
        ax.scatter(
            sub["x"],
            sub["Log2FC"],
            s=9,
            alpha=0.55,
            linewidths=0,
            color=sample_colors[sample],
            label=sample,
            rasterized=True,
            zorder=3,
        )

    ax.axhline(0, color="0.25", linestyle="--", linewidth=1.0, zorder=2)
    ax.yaxis.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(dot_df["x_base"].min() - 1, dot_df["x_base"].max() + 1)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xticks([])
    ax.set_ylabel("Log2 fold-change vs K", fontsize=11)
    ax.set_xlabel("Barcodes", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for _, info in group_df.iterrows():
        label = str(info["pMHC"]).replace("_", "\n", 1) + f"\n(n={int(info['n_barcodes'])})"
        ax.text(
            info["center"],
            -0.12,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.legend(
        frameon=False,
        ncol=len(sample_order),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22)

    png_out = FIGURE_DIR / "barcode_log2FC_dotplot_J_vs_K.png"
    pdf_out = FIGURE_DIR / "barcode_log2FC_dotplot_J_vs_K.pdf"
    fig.savefig(png_out, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return {"png": relpath(png_out), "pdf": relpath(pdf_out)}


def build_violin_tables(barcode_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    group_rows = []
    sample_gap = 1.6
    x = 1.0

    for sample_index, sample in enumerate(SAMPLE_ORDER):
        start_x = x
        group_has_data = False
        lfc_col = f"{sample}_log2FC_vs_K"
        if lfc_col not in barcode_df.columns:
            continue

        for pmhc in PMHC_ORDER:
            vals = barcode_df.loc[barcode_df["pMHC"] == pmhc, ["Barcode", lfc_col]].copy()
            vals[lfc_col] = (
                pd.to_numeric(vals[lfc_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            vals = vals.dropna(subset=[lfc_col])
            if vals.empty:
                continue

            for _, row in vals.iterrows():
                rows.append(
                    {
                        "Sample": sample,
                        "pMHC": pmhc,
                        "peptide_label": PEPTIDE_LABELS[pmhc],
                        "Barcode": row["Barcode"],
                        "Log2FC": float(row[lfc_col]),
                        "position": x,
                    }
                )
            x += 1.0
            group_has_data = True

        if group_has_data:
            end_x = x - 1.0
            group_rows.append(
                {
                    "sample": sample,
                    "start": start_x - 0.5,
                    "end": end_x + 0.5,
                    "center": (start_x + end_x) / 2,
                }
            )
            if sample_index < len(SAMPLE_ORDER) - 1:
                x += sample_gap

    violin_df = pd.DataFrame(rows)
    group_df = pd.DataFrame(group_rows)
    if violin_df.empty:
        raise ValueError("No barcode-level log2FC data found for violin plot.")

    violin_df.to_csv(FIGURE_TABLE_DIR / "pmhc_log2FC_violin_by_sample_table.csv", index=False)
    group_df.to_csv(FIGURE_TABLE_DIR / "pmhc_log2FC_violin_by_sample_groups.csv", index=False)
    return violin_df, group_df


def plot_violin(
    violin_df: pd.DataFrame, group_df: pd.DataFrame, summary_df: pd.DataFrame
) -> dict[str, str]:
    position_meta = (
        violin_df[["position", "Sample", "pMHC", "peptide_label"]]
        .drop_duplicates()
        .sort_values("position")
        .reset_index(drop=True)
    )
    positions = position_meta["position"].to_list()
    datasets = [
        violin_df.loc[violin_df["position"] == pos, "Log2FC"].to_numpy(dtype=float)
        for pos in positions
    ]

    all_vals = np.concatenate(datasets)
    all_vals = all_vals[np.isfinite(all_vals)]
    abs_max = np.nanmax(np.abs(all_vals))
    y_pad = max(0.4, abs_max * 0.08)
    y_lim = abs_max + y_pad

    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=300)
    vp = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in vp["bodies"]:
        body.set_facecolor("0.80")
        body.set_edgecolor("0.35")
        body.set_linewidth(0.8)
        body.set_alpha(1.0)

    for _, meta in position_meta.iterrows():
        row = summary_df.loc[
            (summary_df["Sample"] == meta["Sample"]) & (summary_df["pMHC"] == meta["pMHC"])
        ]
        if row.empty:
            continue
        row = row.iloc[0]
        q25 = float(row["barcode_q25_log2FC"])
        med = float(row["barcode_median_log2FC"])
        q75 = float(row["barcode_q75_log2FC"])
        ax.vlines(meta["position"], q25, q75, color="black", linewidth=2.0, zorder=3)
        ax.scatter(
            [meta["position"]],
            [med],
            s=24,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

    for (_, left), (_, right) in zip(group_df.iloc[:-1].iterrows(), group_df.iloc[1:].iterrows()):
        sep_x = (left["end"] + right["start"]) / 2
        ax.axvline(sep_x, color="0.78", linewidth=1.0, zorder=1)

    ax.axhline(0, color="0.25", linestyle="--", linewidth=1.0, zorder=1)
    ax.yaxis.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(min(positions) - 0.8, max(positions) + 0.8)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xticks(positions)
    ax.set_xticklabels(position_meta["peptide_label"].to_list(), fontsize=9)
    ax.set_ylabel("Barcode Log2FC", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for _, meta in group_df.iterrows():
        ax.text(
            meta["center"],
            -0.13,
            meta["sample"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    mapping_text = "   ".join([f"{PEPTIDE_LABELS[p]} = {p}" for p in PMHC_ORDER])
    fig.text(0.5, 0.01, mapping_text, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.22)

    png_out = FIGURE_DIR / "pmhc_log2FC_violin_by_sample.png"
    pdf_out = FIGURE_DIR / "pmhc_log2FC_violin_by_sample.pdf"
    fig.savefig(png_out, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return {"png": relpath(png_out), "pdf": relpath(pdf_out)}


def output_file_records() -> list[dict[str, Any]]:
    output_roots = [INTERMEDIATE_DIR, FIGURE_TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]
    records = []
    for root in output_roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "run_summary.json":
                records.append(
                    {
                        "relative_path": relpath(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return records


def main() -> None:
    start_time = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    raw_manifest = write_raw_manifest()
    _, filtered_barcodes, _, barcode_qc = generate_barcodes()
    count_df, sample_qc = build_count_table(filtered_barcodes)
    barcode_df, summary_df, k_scaling_df, normalization_qc = compute_log2fc(count_df)
    dot_df, dot_group_df = build_dotplot_tables(barcode_df)
    dot_outputs = plot_dotplot(dot_df, dot_group_df)
    violin_df, violin_group_df = build_violin_tables(barcode_df)
    violin_outputs = plot_violin(violin_df, violin_group_df, summary_df)

    qc_summary = {
        "barcode_generation": barcode_qc,
        "fastq_counting": sample_qc,
        "normalization": normalization_qc,
        "tables": {
            "pmhc_barcode_counts_rows": int(len(count_df)),
            "pmhc_barcode_J_vs_K_log2FC_rows": int(len(barcode_df)),
            "pmhc_pMHC_summary_J_vs_K_rows": int(len(summary_df)),
            "K_equalization_summary_rows": int(len(k_scaling_df)),
            "barcode_log2FC_dotplot_rows": int(len(dot_df)),
            "pmhc_log2FC_violin_rows": int(len(violin_df)),
        },
        "figures": {
            "barcode_log2FC_dotplot_J_vs_K": dot_outputs,
            "pmhc_log2FC_violin_by_sample": violin_outputs,
        },
        "notes": [
            "FASTQ reads are searched for reverse-complement barcode sequences.",
            "FCS files are copied into the package for raw-folder provenance but are not used by this workflow.",
        ],
    }
    write_json(SUMMARY_DIR / "qc_summary.json", qc_summary)

    run_summary = {
        "workflow": "Figure 1F pMHC barcode J-vs-K analysis and figure generation",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - start_time, 3),
        "root": str(ROOT),
        "script": relpath(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "matplotlib": mpl.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "raw_manifest": raw_manifest,
        "outputs": output_file_records(),
    }
    write_json(SUMMARY_DIR / "run_summary.json", run_summary)

    print(f"Workflow complete: {ROOT}")
    print(f"Run summary: {SUMMARY_DIR / 'run_summary.json'}")
    print(f"QC summary:  {SUMMARY_DIR / 'qc_summary.json'}")


if __name__ == "__main__":
    main()
