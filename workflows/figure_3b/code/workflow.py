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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FASTQ_DIR = RAW_DIR / "fastq"
REFERENCE_DIR = RAW_DIR / "reference"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
FIGURE_TABLE_DIR = ROOT / "data" / "figure_tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
SUMMARY_DIR = ROOT / "outputs" / "summaries"

for directory in [INTERMEDIATE_DIR, FIGURE_TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

matplotlib_cache = Path(tempfile.gettempdir()) / "figure_3b_matplotlib_cache"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

LIBRARY_FILE = REFERENCE_DIR / "Library.txt"
COMPARE_FILE = REFERENCE_DIR / "Compare.txt"
COUNTS_FILE = INTERMEDIATE_DIR / "counts.csv"
LFC_FILE = FIGURE_TABLE_DIR / "lfc.csv"

MAX_LOWER_MISMATCHES = 3
PSEUDOCOUNT = 1.0
PLOT_SAMPLE_X = "NT400-R1-1E4X"
PLOT_SAMPLE_Y = "NT400-R2-1E4X"

META_COLS = ["ID", "BC", "pMHC"]
PMHC_FIXED_COLORS = {
    "NC": "#D6D3D1",
    "B0801": "#FDA4AF",
    "A0201": "#67E8F9",
    "A0201-NYESO1": "#FDBA74",
    "A0201-MART1": "#7FA3FF",
}
PMHC_LEGEND_ORDER = ["NC", "B0801", "A0201", "A0201-NYESO1", "A0201-MART1"]


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
            role = "sequencing_fastq"
        elif path.name in {"Library.txt", "Compare.txt"}:
            role = "reference_table"
        else:
            role = "raw_input"
        entries.append(
            {
                "relative_path": relpath(path),
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    pd.DataFrame(entries).to_csv(SUMMARY_DIR / "raw_data_manifest.csv", index=False)
    return entries


def load_library(path: Path = LIBRARY_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [column.strip() for column in df.columns]
    missing = [column for column in META_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    df = df[META_COLS].copy()
    df["BC"] = df["BC"].astype(str).str.strip()
    df["pMHC"] = df["pMHC"].astype(str).str.strip()
    return df


def parse_barcode_pattern(barcode: str):
    if not barcode:
        raise ValueError("Encountered an empty barcode.")

    upper_positions = []
    upper_bases = []
    lower_positions = []
    lower_bases = []
    for index, character in enumerate(barcode):
        if not character.isalpha():
            raise ValueError(f"Barcode {barcode!r} has non-letter character at {index}.")
        base = character.upper()
        if character.isupper():
            upper_positions.append(index)
            upper_bases.append(base)
        else:
            lower_positions.append(index)
            lower_bases.append(base)

    return (
        len(barcode),
        tuple(upper_positions),
        "".join(upper_bases),
        tuple(lower_positions),
        "".join(lower_bases),
    )


def build_match_lookup(library_df: pd.DataFrame):
    library_df = library_df.copy()
    parsed = library_df["BC"].apply(parse_barcode_pattern)
    library_df["_BC_LEN"] = [entry[0] for entry in parsed]
    library_df["_UPPER_POS"] = [entry[1] for entry in parsed]
    library_df["_UPPER_BASES"] = [entry[2] for entry in parsed]
    library_df["_LOWER_POS"] = [entry[3] for entry in parsed]
    library_df["_LOWER_BASES"] = [entry[4] for entry in parsed]

    upper_lookup = defaultdict(lambda: defaultdict(list))
    upper_max_pos = {}
    for row_idx, row in library_df.iterrows():
        upper_positions = row["_UPPER_POS"]
        upper_bases = row["_UPPER_BASES"]
        upper_lookup[upper_positions][upper_bases].append(
            (row_idx, row["_BC_LEN"], row["_LOWER_POS"], row["_LOWER_BASES"])
        )
        upper_max_pos[upper_positions] = upper_positions[-1] if upper_positions else -1

    upper_masks = tuple(sorted(upper_lookup.keys(), key=lambda item: (len(item), item)))
    upper_lookup = {key: dict(value) for key, value in upper_lookup.items()}
    return library_df, upper_lookup, upper_max_pos, upper_masks


def fastq_sample_base(path: Path) -> str:
    name = path.name
    if name.endswith(".fastq.gz"):
        stem = name[:-9]
    elif name.endswith(".fq.gz"):
        stem = name[:-6]
    else:
        stem = path.stem
    return stem.split("_")[0]


def assign_sample_columns(fastq_files: list[Path]) -> tuple[dict[Path, str], dict[str, list[str]]]:
    base_names = [fastq_sample_base(path) for path in fastq_files]
    totals = Counter(base_names)
    seen: Counter[str] = Counter()
    sample_columns: dict[Path, str] = {}
    duplicate_groups: dict[str, list[str]] = defaultdict(list)

    for path, base in zip(fastq_files, base_names):
        seen[base] += 1
        if totals[base] > 1:
            sample = f"{base}-{seen[base]}"
            duplicate_groups[base].append(sample)
        else:
            sample = base
        sample_columns[path] = sample

    return sample_columns, dict(duplicate_groups)


def count_barcodes_in_fastq(
    fastq_path: Path,
    upper_lookup: dict,
    upper_max_pos: dict,
    upper_masks: tuple,
    n_library_rows: int,
    max_lower_mismatches: int = MAX_LOWER_MISMATCHES,
) -> tuple[np.ndarray, dict[str, Any]]:
    counts = np.zeros(n_library_rows, dtype=np.int64)
    reads_scanned = 0
    reads_with_match = 0

    with gzip.open(fastq_path, "rt") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            seq = handle.readline()
            plus = handle.readline()
            qual = handle.readline()

            if not (seq and plus and qual):
                raise ValueError(f"Incomplete FASTQ record found in {fastq_path}")

            reads_scanned += 1
            seq = seq.strip().upper()
            seq_len = len(seq)
            read_matched = False

            for upper_positions in upper_masks:
                if seq_len <= upper_max_pos[upper_positions]:
                    continue

                upper_key = "".join(seq[position] for position in upper_positions)
                candidates = upper_lookup[upper_positions].get(upper_key)
                if not candidates:
                    continue

                for row_idx, required_len, lower_positions, lower_bases in candidates:
                    if seq_len < required_len:
                        continue

                    mismatches = 0
                    for position, base in zip(lower_positions, lower_bases):
                        if seq[position] != base:
                            mismatches += 1
                            if mismatches > max_lower_mismatches:
                                break

                    if mismatches <= max_lower_mismatches:
                        counts[row_idx] += 1
                        read_matched = True

            reads_with_match += int(read_matched)

    qc = {
        "fastq": relpath(fastq_path),
        "reads_scanned": reads_scanned,
        "reads_with_at_least_one_barcode_match": reads_with_match,
        "total_barcode_matches": int(counts.sum()),
    }
    return counts, qc


def build_counts_table(library_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    parsed_library, upper_lookup, upper_max_pos, upper_masks = build_match_lookup(library_df)
    fastq_files = sorted(FASTQ_DIR.glob("*.fastq.gz"), key=lambda path: path.name)
    if not fastq_files:
        raise FileNotFoundError(f"No FASTQ files found in {FASTQ_DIR}")

    sample_columns, duplicate_groups = assign_sample_columns(fastq_files)
    output_parts = [parsed_library[META_COLS].reset_index(drop=True)]
    fastq_qc = []

    for fastq_file in fastq_files:
        sample = sample_columns[fastq_file]
        counts, qc = count_barcodes_in_fastq(
            fastq_path=fastq_file,
            upper_lookup=upper_lookup,
            upper_max_pos=upper_max_pos,
            upper_masks=upper_masks,
            n_library_rows=len(parsed_library),
        )
        output_parts.append(pd.Series(counts, name=sample))
        qc["sample_column"] = sample
        fastq_qc.append(qc)

    counts_df = pd.concat(output_parts, axis=1)
    for combined_name, component_columns in duplicate_groups.items():
        counts_df[combined_name] = counts_df[component_columns].sum(axis=1)

    counts_df.to_csv(COUNTS_FILE, index=False)
    lookup_qc = {
        "library_rows": int(len(parsed_library)),
        "unique_uppercase_masks": int(len(upper_masks)),
        "max_lower_mismatches": MAX_LOWER_MISMATCHES,
        "duplicate_sample_groups": duplicate_groups,
    }
    return counts_df, fastq_qc, lookup_qc


def load_compare(path: Path = COMPARE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [column.strip() for column in df.columns]
    missing = [column for column in ["S", "R"] if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    df = df[["S", "R"]].copy()
    df["S"] = df["S"].astype(str).str.strip()
    df["R"] = df["R"].astype(str).str.strip()
    return df


def compute_lfc(counts_df: pd.DataFrame, compare_df: pd.DataFrame) -> pd.DataFrame:
    sample_cols = [column for column in counts_df.columns if column not in META_COLS]
    missing_samples = sorted(set(compare_df["S"]) - set(sample_cols))
    missing_refs = sorted(set(compare_df["R"]) - set(sample_cols))
    if missing_samples:
        raise ValueError(f"S sample(s) not found in counts table: {missing_samples}")
    if missing_refs:
        raise ValueError(f"R reference sample(s) not found in counts table: {missing_refs}")

    mapping_check = compare_df.groupby("S")["R"].nunique()
    conflicting = mapping_check[mapping_check > 1].index.tolist()
    if conflicting:
        raise ValueError(f"Conflicting S-to-R mappings found: {conflicting}")

    compare_unique = compare_df.drop_duplicates(subset=["S"], keep="first").reset_index(drop=True)
    counts_numeric = counts_df.copy()
    for column in sample_cols:
        counts_numeric[column] = pd.to_numeric(counts_numeric[column], errors="raise")

    lfc_df = counts_numeric[META_COLS].copy()
    for _, row in compare_unique.iterrows():
        sample = row["S"]
        reference = row["R"]
        lfc_df[sample] = np.log2(
            (counts_numeric[sample] + PSEUDOCOUNT)
            / (counts_numeric[reference] + PSEUDOCOUNT)
        )

    lfc_df.to_csv(LFC_FILE, index=False)
    return lfc_df


def pmhc_order(df: pd.DataFrame) -> list[str]:
    observed = list(dict.fromkeys(df["pMHC"].tolist()))
    ordered = [pmhc for pmhc in PMHC_LEGEND_ORDER if pmhc in observed]
    ordered.extend([pmhc for pmhc in observed if pmhc not in ordered])
    return ordered


def pmhc_color_map(order: list[str]) -> dict[str, str]:
    color_map = {pmhc: PMHC_FIXED_COLORS[pmhc] for pmhc in order if pmhc in PMHC_FIXED_COLORS}
    remaining = [pmhc for pmhc in order if pmhc not in color_map]
    if remaining:
        cmap = plt.get_cmap("tab20", len(remaining))
        for index, pmhc in enumerate(remaining):
            color_map[pmhc] = cmap(index)
    return color_map


def legend_handles(order: list[str], color_map: dict[str, str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            label=str(pmhc),
            markerfacecolor=color_map[pmhc],
            markeredgecolor=color_map[pmhc],
        )
        for pmhc in order
    ]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))


def build_plot_table(lfc_df: pd.DataFrame, sample_x: str, sample_y: str) -> pd.DataFrame:
    for sample in [sample_x, sample_y]:
        if sample not in lfc_df.columns:
            sample_cols = [column for column in lfc_df.columns if column not in META_COLS]
            raise ValueError(f"Sample {sample!r} not found in LFC table. Available: {sample_cols}")

    order = pmhc_order(lfc_df)
    colors = pmhc_color_map(order)
    plot_df = lfc_df[META_COLS + [sample_x, sample_y]].copy()
    plot_df = plot_df.rename(columns={sample_x: "lfc_x", sample_y: "lfc_y"})
    plot_df["sample_x"] = sample_x
    plot_df["sample_y"] = sample_y
    plot_df["color"] = plot_df["pMHC"].map(colors)
    plot_df.to_csv(FIGURE_TABLE_DIR / "lfc_NT400-R1-1E4X_vs_NT400-R2-1E4X_table.csv", index=False)
    return plot_df


def plot_lfc_scatter(plot_df: pd.DataFrame, sample_x: str, sample_y: str) -> dict[str, str]:
    order = pmhc_order(plot_df)
    colors = pmhc_color_map(order)

    fig, ax = plt.subplots(figsize=(8, 8))
    for pmhc in order:
        group = plot_df.loc[plot_df["pMHC"] == pmhc]
        if group.empty:
            continue
        ax.scatter(
            group["lfc_x"],
            group["lfc_y"],
            s=24,
            alpha=0.8,
            label=str(pmhc),
            color=colors[pmhc],
        )

    ax.set_xlabel(f"LFC ({sample_x})")
    ax.set_ylabel(f"LFC ({sample_y})")
    ax.set_title(f"LFC comparison: {sample_x} vs {sample_y}")
    ax.set_box_aspect(1)
    ax.legend(
        handles=legend_handles(order, colors),
        title="pMHC",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )
    fig.tight_layout()

    base = f"lfc_{safe_name(sample_x)}_vs_{safe_name(sample_y)}"
    png_path = FIGURE_DIR / f"{base}.png"
    pdf_path = FIGURE_DIR / f"{base}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": relpath(png_path), "pdf": relpath(pdf_path)}


def output_file_records() -> list[dict[str, Any]]:
    records = []
    for root in [INTERMEDIATE_DIR, FIGURE_TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]:
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
    library_df = load_library()
    counts_df, fastq_qc, lookup_qc = build_counts_table(library_df)
    compare_df = load_compare()
    lfc_df = compute_lfc(counts_df, compare_df)
    plot_df = build_plot_table(lfc_df, PLOT_SAMPLE_X, PLOT_SAMPLE_Y)
    figure_outputs = plot_lfc_scatter(plot_df, PLOT_SAMPLE_X, PLOT_SAMPLE_Y)

    qc_summary = {
        "parameters": {
            "max_lowercase_barcode_mismatches": MAX_LOWER_MISMATCHES,
            "lfc_pseudocount": PSEUDOCOUNT,
            "plot_sample_x": PLOT_SAMPLE_X,
            "plot_sample_y": PLOT_SAMPLE_Y,
        },
        "library": {
            **lookup_qc,
            "pMHC_counts": library_df["pMHC"].value_counts().sort_index().to_dict(),
        },
        "fastq_counting": fastq_qc,
        "tables": {
            "counts_rows": int(len(counts_df)),
            "counts_columns": int(len(counts_df.columns)),
            "lfc_rows": int(len(lfc_df)),
            "lfc_columns": int(len(lfc_df.columns)),
            "plot_table_rows": int(len(plot_df)),
        },
        "figures": {
            f"lfc_{PLOT_SAMPLE_X}_vs_{PLOT_SAMPLE_Y}": figure_outputs,
        },
    }
    write_json(SUMMARY_DIR / "qc_summary.json", qc_summary)

    run_summary = {
        "workflow": "Figure 3B barcode count, LFC, and scatter figure generation",
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
