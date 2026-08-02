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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

matplotlib_cache = Path(tempfile.gettempdir()) / "figure_2i_matplotlib_cache"
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

PBC_CSV = REFERENCE_DIR / "PBC.csv"
REF_CSV = REFERENCE_DIR / "hCRISPRi_gRNA-BC.csv"

MIN_READ_LEN = 930
FIRST_WINDOW = 50
LAST_WINDOW = 50
OFFSETS = {
    "BC": (20, 50),
    "gRNA1": (329, 369),
    "gRNA2": (759, 799),
    "UMI": (40, 74),
}

MATCH_COLUMNS = ["ReadID", "PBC1", "PBC2", "UMI", "BC", "gRNA1", "gRNA2"]
COLLAPSED_COLUMNS = ["UMI", "ReadCount", "PBC1", "PBC2", "BC", "gRNA1", "gRNA2"]
RC_TRANS = str.maketrans("ACGTNacgtn", "TGCANtgcan")

COUPLING_CATEGORIES = [
    {
        "column": "prop_matching_BC_triplet",
        "count_column": "umis_matching_BC_triplet",
        "label": "BC==gRNA1==gRNA2",
        "color": "#1f77b4",
    },
    {
        "column": "prop_exactly_one_gRNA_matching_BC",
        "count_column": "umis_exactly_one_gRNA_matching_BC",
        "label": "Exactly one gRNA matches BC",
        "color": "#ff7f0e",
    },
    {
        "column": "prop_gRNA1_eq_gRNA2_not_BC",
        "count_column": "umis_gRNA1_eq_gRNA2_not_BC",
        "label": "gRNA1==gRNA2!=BC",
        "color": "#2ca02c",
    },
    {
        "column": "prop_other",
        "count_column": "umis_other",
        "label": "Other",
        "color": "#d62728",
    },
]


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
        elif path.name in {"PBC.csv", "hCRISPRi_gRNA-BC.csv"}:
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


def sanitize_seq(value: str) -> str:
    return re.sub(r"\s+", "", str(value).upper())


def revcomp(seq: str) -> str:
    return seq.translate(RC_TRANS)[::-1]


def base_id(value: str) -> str:
    return re.sub(r"_(\d+)$", "", str(value))


def read_fastq(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            seq = handle.readline()
            plus = handle.readline()
            qual = handle.readline()
            if not seq or not plus or not qual:
                raise ValueError(f"Malformed FASTQ record in {path}")
            if not header.startswith("@"):
                continue
            read_id = header[1:].strip().split()[0]
            yield read_id, sanitize_seq(seq)


def load_pbc_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = {"PBC_ID", "PBC1", "PBC2"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")
    df["PBC1"] = df["PBC1"].map(sanitize_seq)
    df["PBC2"] = df["PBC2"].map(sanitize_seq)
    return df


def build_pbc_maps(pbc_df: pd.DataFrame) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, int]]:
    def unique_pairs(column: str) -> tuple[list[tuple[str, str]], int]:
        buckets: dict[str, set[str]] = defaultdict(set)
        for _, row in pbc_df.iterrows():
            seq = row[column]
            if seq:
                buckets[seq].add(row["PBC_ID"])

        pairs = []
        ambiguous = 0
        for seq, labels in buckets.items():
            if len(labels) == 1:
                pairs.append((seq, next(iter(labels))))
            else:
                ambiguous += 1
        pairs.sort(key=lambda item: -len(item[0]))
        return pairs, ambiguous

    pbc1_list, pbc1_ambiguous = unique_pairs("PBC1")
    pbc2_list, pbc2_ambiguous = unique_pairs("PBC2")
    qc = {
        "PBC1_unique_sequences": len(pbc1_list),
        "PBC2_unique_sequences": len(pbc2_list),
        "PBC1_ambiguous_sequences_removed": pbc1_ambiguous,
        "PBC2_ambiguous_sequences_removed": pbc2_ambiguous,
    }
    return pbc1_list, pbc2_list, qc


def load_ref_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = {"ID", "gRNA1", "gRNA2", "BC"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")
    for column in ["ID", "gRNA1", "gRNA2", "BC"]:
        df[column] = df[column].astype(str)
    df["gRNA1"] = df["gRNA1"].map(sanitize_seq)
    df["gRNA2"] = df["gRNA2"].map(sanitize_seq)
    df["BC"] = df["BC"].map(sanitize_seq)
    return df


def build_sequence_to_baseid_maps(
    ref_df: pd.DataFrame,
) -> tuple[
    dict[int, dict[str, str]],
    dict[int, dict[str, str]],
    dict[int, dict[str, str]],
    dict[str, int],
]:
    def build(column: str) -> tuple[dict[int, dict[str, str]], int, int]:
        buckets: dict[str, set[str]] = defaultdict(set)
        for _, row in ref_df.iterrows():
            seq = row[column]
            if seq:
                buckets[seq].add(base_id(row["ID"]))

        mapping: dict[str, str] = {}
        ambiguous = 0
        for seq, ids in buckets.items():
            if len(ids) == 1:
                mapping[seq] = next(iter(ids))
            else:
                ambiguous += 1
        lookup: dict[int, dict[str, str]] = defaultdict(dict)
        for seq, base in mapping.items():
            lookup[len(seq)][seq] = base
        return dict(lookup), ambiguous, len(mapping)

    bc_map, bc_ambiguous, bc_unique = build("BC")
    g1_map, g1_ambiguous, g1_unique = build("gRNA1")
    g2_map, g2_ambiguous, g2_unique = build("gRNA2")
    qc = {
        "reference_rows": int(len(ref_df)),
        "BC_unique_sequences": bc_unique,
        "gRNA1_unique_sequences": g1_unique,
        "gRNA2_unique_sequences": g2_unique,
        "BC_ambiguous_sequences_removed": bc_ambiguous,
        "gRNA1_ambiguous_sequences_removed": g1_ambiguous,
        "gRNA2_ambiguous_sequences_removed": g2_ambiguous,
    }
    return bc_map, g1_map, g2_map, qc


def choose_match_in_segment(
    segment: str, candidates: list[tuple[str, str]]
) -> tuple[str, int, str] | None:
    hits = []
    for seq, label in candidates:
        pos = segment.find(seq)
        if pos != -1:
            hits.append((seq, label, pos, len(seq)))
    if not hits:
        return None
    hits.sort(key=lambda item: (-item[3], item[2]))
    seq, label, pos, length = hits[0]
    return label, pos + length, seq


def find_pbc_hits(
    seq: str,
    pbc1_list: list[tuple[str, str]],
    pbc2_list: list[tuple[str, str]],
) -> tuple[str, str, str, int, str] | None:
    def attempt(candidate_seq: str) -> tuple[str, str, int] | None:
        prefix = candidate_seq[:FIRST_WINDOW]
        suffix = candidate_seq[-LAST_WINDOW:] if LAST_WINDOW <= len(candidate_seq) else candidate_seq
        pbc1 = choose_match_in_segment(prefix, pbc1_list)
        pbc2 = choose_match_in_segment(suffix, pbc2_list)
        if not pbc1 or not pbc2:
            return None
        pbc1_id, pbc1_end, _ = pbc1
        pbc2_id, _, _ = pbc2
        return pbc1_id, pbc2_id, pbc1_end

    hit = attempt(seq)
    if hit:
        pbc1_id, pbc2_id, pbc1_end = hit
        return seq, pbc1_id, pbc2_id, pbc1_end, "fwd"

    rc_seq = revcomp(seq)
    hit = attempt(rc_seq)
    if hit:
        pbc1_id, pbc2_id, pbc1_end = hit
        return rc_seq, pbc1_id, pbc2_id, pbc1_end, "rev"

    return None


def slice_window(seq: str, start_inclusive: int, end_inclusive: int) -> str:
    start = max(0, start_inclusive)
    if end_inclusive < 0 or start >= len(seq):
        return ""
    return seq[start : min(end_inclusive + 1, len(seq))]


def find_any_in_window(
    window: str, seq_lookup: dict[int, dict[str, str]]
) -> tuple[str, str] | None:
    for length in sorted(seq_lookup, reverse=True):
        if length <= 0 or len(window) < length:
            continue
        matches = []
        lookup = seq_lookup[length]
        for start in range(len(window) - length + 1):
            subseq = window[start : start + length]
            base = lookup.get(subseq)
            if base is not None:
                matches.append((subseq, base, start))
        if not matches:
            continue
        if len({match[1] for match in matches}) > 1:
            return None
        matches.sort(key=lambda item: item[2])
        return matches[0][1], matches[0][0]
    return None


def find_umi_16nt(window: str) -> str | None:
    match = re.search(r"GAAG([ACGT]{16})TGAA", window)
    return match.group(1) if match else None


def fastq_stub(path: Path) -> str:
    name = path.name
    if name.endswith(".fastq.gz"):
        return name[:-9]
    if name.endswith(".fq.gz"):
        return name[:-6]
    if name.endswith(".fastq"):
        return name[:-6]
    if name.endswith(".fq"):
        return name[:-3]
    return path.stem


def proportion(num: int | float, den: int | float) -> float:
    return (num / den) if den else 0.0


def analyze_fastq_file(
    fq_path: Path,
    pbc1_list: list[tuple[str, str]],
    pbc2_list: list[tuple[str, str]],
    bc_map: dict[int, dict[str, str]],
    g1_map: dict[int, dict[str, str]],
    g2_map: dict[int, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_reads = 0
    len_pass = 0
    pbc_forward = 0
    pbc_reverse = 0
    pbc_total = 0
    bc_found_after_pbc = 0
    g1_found_after_pbc = 0
    g2_found_after_pbc = 0
    triple_feature_found = 0
    umi_found_on_triple = 0
    rows = []

    for read_id, seq in read_fastq(fq_path):
        total_reads += 1
        if len(seq) <= MIN_READ_LEN:
            continue
        len_pass += 1

        hit = find_pbc_hits(seq, pbc1_list, pbc2_list)
        if not hit:
            continue

        used_seq, pbc1_id, pbc2_id, pbc1_end_idx, orientation = hit
        pbc_forward += int(orientation == "fwd")
        pbc_reverse += int(orientation == "rev")
        pbc_total += 1

        def window(key: str) -> str:
            offset_start, offset_end = OFFSETS[key]
            return slice_window(used_seq, pbc1_end_idx + offset_start, pbc1_end_idx + offset_end)

        bc_hit = find_any_in_window(window("BC"), bc_map)
        g1_hit = find_any_in_window(window("gRNA1"), g1_map)
        g2_hit = find_any_in_window(window("gRNA2"), g2_map)

        bc_found_after_pbc += int(bc_hit is not None)
        g1_found_after_pbc += int(g1_hit is not None)
        g2_found_after_pbc += int(g2_hit is not None)

        if not (bc_hit and g1_hit and g2_hit):
            continue
        triple_feature_found += 1

        umi = find_umi_16nt(window("UMI"))
        if not umi:
            continue
        umi_found_on_triple += 1

        rows.append(
            {
                "ReadID": read_id,
                "PBC1": pbc1_id,
                "PBC2": pbc2_id,
                "UMI": umi,
                "BC": bc_hit[0],
                "gRNA1": g1_hit[0],
                "gRNA2": g2_hit[0],
            }
        )

    matches_df = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    summary = {
        "file": fq_path.name,
        "total_reads": total_reads,
        "reads_len_gt_930": len_pass,
        "pbc_forward_count": pbc_forward,
        "pbc_reverse_count": pbc_reverse,
        "pbc_total_count": pbc_total,
        "pbc_total_prop_of_len_pass": proportion(pbc_total, len_pass),
        "BC_match_after_pbc": bc_found_after_pbc,
        "BC_match_prop_of_pbc": proportion(bc_found_after_pbc, pbc_total),
        "gRNA1_match_after_pbc": g1_found_after_pbc,
        "gRNA1_match_prop_of_pbc": proportion(g1_found_after_pbc, pbc_total),
        "gRNA2_match_after_pbc": g2_found_after_pbc,
        "gRNA2_match_prop_of_pbc": proportion(g2_found_after_pbc, pbc_total),
        "triple_feature_count": triple_feature_found,
        "triple_feature_prop_of_pbc": proportion(triple_feature_found, pbc_total),
        "umi_on_triple_count": umi_found_on_triple,
        "umi_on_triple_prop_of_triple": proportion(umi_found_on_triple, triple_feature_found),
        "final_recorded_rows": len(matches_df),
        "final_prop_of_len_pass": proportion(len(matches_df), len_pass),
    }
    return matches_df, pd.DataFrame([summary])


def analyze_worker(args: tuple[Any, ...]) -> dict[str, Any]:
    fq_path, pbc1_list, pbc2_list, bc_map, g1_map, g2_map = args
    fq_path = Path(fq_path)
    matches_df, summary_df = analyze_fastq_file(
        fq_path, pbc1_list, pbc2_list, bc_map, g1_map, g2_map
    )
    stub = fastq_stub(fq_path)
    matches_path = INTERMEDIATE_DIR / f"{stub}_matches.csv"
    summary_path = INTERMEDIATE_DIR / f"{stub}_summary.csv"
    matches_df.to_csv(matches_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    summary = summary_df.iloc[0].to_dict()
    summary["matches_path"] = relpath(matches_path)
    summary["summary_path"] = relpath(summary_path)
    return summary


def discover_fastqs() -> list[Path]:
    patterns = ["*.fastq", "*.fq", "*.fastq.gz", "*.fq.gz"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(FASTQ_DIR.glob(pattern))
    return sorted(paths, key=lambda path: path.name)


def run_fastq_matching(
    pbc1_list: list[tuple[str, str]],
    pbc2_list: list[tuple[str, str]],
    bc_map: dict[int, dict[str, str]],
    g1_map: dict[int, dict[str, str]],
    g2_map: dict[int, dict[str, str]],
) -> pd.DataFrame:
    fastq_files = discover_fastqs()
    if not fastq_files:
        raise FileNotFoundError(f"No FASTQ files found in {FASTQ_DIR}")

    worker_count = int(os.environ.get("FIG2I_WORKERS", min(4, len(fastq_files))))
    worker_count = max(1, min(worker_count, len(fastq_files)))
    backend = os.environ.get("FIG2I_PARALLEL", "thread").strip().lower()
    if backend not in {"process", "thread", "serial"}:
        backend = "process"

    args = [
        (path, pbc1_list, pbc2_list, bc_map, g1_map, g2_map)
        for path in fastq_files
    ]

    summaries: list[dict[str, Any]] = []
    if backend == "serial" or worker_count == 1:
        for item in args:
            summaries.append(analyze_worker(item))
    else:
        executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        try:
            with executor_cls(max_workers=worker_count) as executor:
                futures = {executor.submit(analyze_worker, item): item[0] for item in args}
                for future in as_completed(futures):
                    summaries.append(future.result())
        except (OSError, PermissionError) as exc:
            if backend != "process":
                raise
            print(f"Process parallelism unavailable ({exc}); retrying with threads.")
            summaries = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(analyze_worker, item): item[0] for item in args}
                for future in as_completed(futures):
                    summaries.append(future.result())

    summary_df = pd.DataFrame(summaries).sort_values("file").reset_index(drop=True)
    output_cols = [
        "file",
        "total_reads",
        "reads_len_gt_930",
        "pbc_forward_count",
        "pbc_reverse_count",
        "pbc_total_count",
        "pbc_total_prop_of_len_pass",
        "BC_match_after_pbc",
        "BC_match_prop_of_pbc",
        "gRNA1_match_after_pbc",
        "gRNA1_match_prop_of_pbc",
        "gRNA2_match_after_pbc",
        "gRNA2_match_prop_of_pbc",
        "triple_feature_count",
        "triple_feature_prop_of_pbc",
        "umi_on_triple_count",
        "umi_on_triple_prop_of_triple",
        "final_recorded_rows",
        "final_prop_of_len_pass",
    ]
    summary_df[output_cols].to_csv(INTERMEDIATE_DIR / "all_files_summary.csv", index=False)
    return summary_df


def load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in MATCH_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    for column in MATCH_COLUMNS:
        df[column] = df[column].astype(str).str.strip()
    return df[df["UMI"] != ""].copy()


def collapse_umis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLLAPSED_COLUMNS)

    species_cols = ["PBC1", "PBC2", "BC", "gRNA1", "gRNA2"]
    rows = []
    for umi, group in df.groupby("UMI", sort=False):
        counts = (
            group.groupby(species_cols, dropna=False, sort=False)
            .size()
            .reset_index(name="n")
        )
        if counts.empty:
            continue
        top_n = counts["n"].max()
        top = counts[counts["n"] == top_n]
        if len(top) > 1:
            continue
        row = top.iloc[0]
        if row["PBC1"] != row["PBC2"]:
            continue
        rows.append(
            {
                "UMI": umi,
                "ReadCount": int(row["n"]),
                "PBC1": row["PBC1"],
                "PBC2": row["PBC2"],
                "BC": row["BC"],
                "gRNA1": row["gRNA1"],
                "gRNA2": row["gRNA2"],
            }
        )

    return pd.DataFrame(rows, columns=COLLAPSED_COLUMNS)


def summarize_coupling(collapsed_df: pd.DataFrame, filename: str) -> pd.DataFrame:
    total_umis = len(collapsed_df)
    if total_umis == 0:
        n_triplet = n_one = n_g12_not_bc = 0
        p_triplet = p_one = p_g12_not_bc = np.nan
    else:
        bc = collapsed_df["BC"]
        g1 = collapsed_df["gRNA1"]
        g2 = collapsed_df["gRNA2"]
        match_triplet = (bc == g1) & (bc == g2)
        match_one = (bc == g1) ^ (bc == g2)
        match_g12_not_bc = (g1 == g2) & (bc != g1)
        n_triplet = int(match_triplet.sum())
        n_one = int(match_one.sum())
        n_g12_not_bc = int(match_g12_not_bc.sum())
        p_triplet = n_triplet / total_umis
        p_one = n_one / total_umis
        p_g12_not_bc = n_g12_not_bc / total_umis

    return pd.DataFrame(
        [
            {
                "file": filename,
                "umis_after_collapse": total_umis,
                "umis_matching_BC_triplet": n_triplet,
                "prop_matching_BC_triplet": p_triplet,
                "umis_exactly_one_gRNA_matching_BC": n_one,
                "prop_exactly_one_gRNA_matching_BC": p_one,
                "umis_gRNA1_eq_gRNA2_not_BC": n_g12_not_bc,
                "prop_gRNA1_eq_gRNA2_not_BC": p_g12_not_bc,
            }
        ]
    )


def run_umi_collapse() -> pd.DataFrame:
    match_paths = sorted(INTERMEDIATE_DIR.glob("*_matches.csv"), key=lambda path: path.name)
    if not match_paths:
        raise FileNotFoundError(f"No *_matches.csv files found in {INTERMEDIATE_DIR}")

    summary_rows = []
    for path in match_paths:
        stub = path.name.replace("_matches.csv", "")
        matches = load_matches(path)
        collapsed = collapse_umis(matches)
        collapsed.to_csv(INTERMEDIATE_DIR / f"{stub}_umi_collapsed.csv", index=False)

        summary = summarize_coupling(collapsed, stub)
        summary.to_csv(INTERMEDIATE_DIR / f"{stub}_umi_coupling_summary.csv", index=False)
        summary_rows.append(summary.iloc[0].to_dict())

    summary_df = pd.DataFrame(summary_rows).sort_values("file").reset_index(drop=True)
    summary_df.to_csv(FIGURE_TABLE_DIR / "all_files_umi_coupling_summary.csv", index=False)
    return summary_df


def build_figure_table(coupling_df: pd.DataFrame) -> pd.DataFrame:
    df = coupling_df.copy()
    used_count_cols = [
        "umis_matching_BC_triplet",
        "umis_exactly_one_gRNA_matching_BC",
        "umis_gRNA1_eq_gRNA2_not_BC",
    ]
    used_prop_cols = [
        "prop_matching_BC_triplet",
        "prop_exactly_one_gRNA_matching_BC",
        "prop_gRNA1_eq_gRNA2_not_BC",
    ]
    df["umis_other"] = df["umis_after_collapse"] - df[used_count_cols].sum(axis=1)
    df["prop_other"] = 1.0 - df[used_prop_cols].sum(axis=1)
    df["sample"] = df["file"].str.extract(r"sample_(\d+)", expand=False).map(lambda x: f"sample_{x}")

    rows = []
    for _, row in df.iterrows():
        for index, category in enumerate(COUPLING_CATEGORIES, start=1):
            rows.append(
                {
                    "sample": row["sample"],
                    "file": row["file"],
                    "category_order": index,
                    "category": category["label"],
                    "count": int(row[category["count_column"]]),
                    "proportion": float(row[category["column"]]),
                    "umis_after_collapse": int(row["umis_after_collapse"]),
                }
            )
    figure_df = pd.DataFrame(rows)
    figure_df.to_csv(FIGURE_TABLE_DIR / "umi_coupling_stacked_bar_table.csv", index=False)
    return figure_df


def plot_umi_coupling(figure_df: pd.DataFrame) -> dict[str, str]:
    samples = (
        figure_df[["sample", "file", "umis_after_collapse"]]
        .drop_duplicates()
        .sort_values("sample")
        .reset_index(drop=True)
    )
    x_positions = np.arange(len(samples))

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bottoms = np.zeros(len(samples))

    for category in COUPLING_CATEGORIES:
        values = []
        for sample in samples["sample"]:
            value = figure_df.loc[
                (figure_df["sample"] == sample) & (figure_df["category"] == category["label"]),
                "proportion",
            ].iloc[0]
            values.append(float(value))
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            width=0.5,
            label=category["label"],
            color=category["color"],
        )
        bottoms += np.array(values)

    ax.set_ylabel("Fraction of collapsed UMIs")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(samples["sample"].tolist())
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
    ax.set_title("UMI coupling categories (post-collapse; PBC1==PBC2)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for index, row in samples.iterrows():
        ax.text(
            x_positions[index],
            1.015,
            f"n={int(row['umis_after_collapse']):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(left=0.08, right=0.78, top=0.86, bottom=0.16)

    png_path = FIGURE_DIR / "umi_coupling_stacked_bar.png"
    pdf_path = FIGURE_DIR / "umi_coupling_stacked_bar.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
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
    pbc_df = load_pbc_table(PBC_CSV)
    pbc1_list, pbc2_list, pbc_qc = build_pbc_maps(pbc_df)
    ref_df = load_ref_table(REF_CSV)
    bc_map, g1_map, g2_map, ref_qc = build_sequence_to_baseid_maps(ref_df)

    fastq_summary = run_fastq_matching(pbc1_list, pbc2_list, bc_map, g1_map, g2_map)
    coupling_summary = run_umi_collapse()
    figure_table = build_figure_table(coupling_summary)
    figure_outputs = plot_umi_coupling(figure_table)

    qc_summary = {
        "parameters": {
            "minimum_read_length_exclusive": MIN_READ_LEN,
            "first_window_nt": FIRST_WINDOW,
            "last_window_nt": LAST_WINDOW,
            "offsets_from_PBC1_end_inclusive": OFFSETS,
            "umi_pattern": "GAAG([ACGT]{16})TGAA",
        },
        "references": {**pbc_qc, **ref_qc},
        "fastq_matching": {
            "files_processed": int(len(fastq_summary)),
            "total_reads": int(fastq_summary["total_reads"].sum()),
            "length_passing_reads": int(fastq_summary["reads_len_gt_930"].sum()),
            "final_recorded_rows": int(fastq_summary["final_recorded_rows"].sum()),
        },
        "umi_coupling": {
            "files_processed": int(len(coupling_summary)),
            "total_collapsed_umis": int(coupling_summary["umis_after_collapse"].sum()),
            "total_matching_BC_triplet": int(coupling_summary["umis_matching_BC_triplet"].sum()),
            "total_exactly_one_gRNA_matching_BC": int(
                coupling_summary["umis_exactly_one_gRNA_matching_BC"].sum()
            ),
            "total_gRNA1_eq_gRNA2_not_BC": int(
                coupling_summary["umis_gRNA1_eq_gRNA2_not_BC"].sum()
            ),
        },
        "tables": {
            "all_files_summary_rows": int(len(fastq_summary)),
            "all_files_umi_coupling_summary_rows": int(len(coupling_summary)),
            "umi_coupling_stacked_bar_rows": int(len(figure_table)),
        },
        "figures": {"umi_coupling_stacked_bar": figure_outputs},
    }
    write_json(SUMMARY_DIR / "qc_summary.json", qc_summary)

    run_summary = {
        "workflow": "Figure 2I UMI coupling analysis and figure generation",
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
