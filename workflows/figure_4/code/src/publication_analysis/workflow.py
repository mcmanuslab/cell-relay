from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import platform
import runpy
import sqlite3
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nt475_publication_mpl"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "figure.dpi": 120,
    }
)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import sparse
from scipy.cluster import hierarchy
from scipy.spatial import cKDTree


STEP_ORDER = [
    "preflight",
    "doublets",
    "barcode_parse",
    "dcbc_identity_precorrection",
    "dcbc_correction",
    "dcbc_identity",
    "classification",
    "read_support_filter",
    "t_cell_metadata",
    "dendritic_cell_metadata",
    "figures",
    "readme",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the NT475 auditable publication workflow.")
    parser.add_argument("--config", default="config.yaml", help="Path to JSON-compatible YAML config.")
    parser.add_argument(
        "--steps",
        nargs="*",
        default=STEP_ORDER,
        choices=STEP_ORDER,
        help="Optional subset of workflow steps to run.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    root = config_path.parent.parent if config_path.parent.name == "code" else config_path.parent
    config = load_config(config_path)
    dirs = ensure_directories(root)
    logger = setup_logging(dirs["logs"])
    logger.info("Starting workflow in %s", root)
    logger.info("Steps: %s", ", ".join(args.steps))

    started = utc_now()
    run_summary: dict[str, Any] = {"started_utc": started, "steps": {}}
    for step in STEP_ORDER:
        if step not in args.steps:
            continue
        t0 = time.time()
        logger.info("STEP START: %s", step)
        if step == "preflight":
            run_preflight(config, root, dirs, logger)
        elif step == "doublets":
            run_doublet_detection(config, root, dirs, logger)
        elif step == "barcode_parse":
            run_barcode_parse(config, root, dirs, logger)
        elif step == "dcbc_identity_precorrection":
            run_dcbc_identity_precorrection(config, root, dirs, logger)
        elif step == "dcbc_correction":
            run_dcbc_correction(config, root, dirs, logger)
        elif step == "dcbc_identity":
            run_dcbc_identity(config, root, dirs, logger)
        elif step == "classification":
            run_cell_classification(config, root, dirs, logger)
        elif step == "read_support_filter":
            run_read_support_filter(config, root, dirs, logger)
        elif step == "t_cell_metadata":
            run_t_cell_metadata(config, root, dirs, logger)
        elif step == "dendritic_cell_metadata":
            run_dendritic_cell_metadata(config, root, dirs, logger)
        elif step == "figures":
            generate_figures(config, root, dirs, logger)
        elif step == "readme":
            write_readme(config, root, dirs, logger)
        elapsed = time.time() - t0
        run_summary["steps"][step] = {"elapsed_seconds": round(elapsed, 3)}
        logger.info("STEP END: %s (%.1f s)", step, elapsed)

    run_summary["finished_utc"] = utc_now()
    summary_name = "workflow_run_summary.json" if list(args.steps) == STEP_ORDER else "workflow_run_summary_last_subset.json"
    write_json(run_summary, dirs["qc"] / summary_name)
    write_software_versions(dirs["qc"] / "software_versions.json")
    logger.info("Workflow complete")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_directories(root: Path) -> dict[str, Path]:
    upstream = root / "data" / "upstream" / "publication_analysis"
    full_summary = root / "outputs" / "summaries" / "full_workflow"
    dirs = {
        "data": upstream / "data_intermediate",
        "tables": upstream / "tables",
        "qc": full_summary,
        "qc_plots": full_summary / "plots",
        "fig_pdf": root / "outputs" / "figures" / "full_workflow" / "pdf",
        "fig_png": root / "outputs" / "figures" / "full_workflow" / "png",
        "logs": full_summary,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("publication_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_dir / "workflow.log", mode="a")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def sha256_file(path: Path, logger: logging.Logger) -> str:
    h = hashlib.sha256()
    total = 0
    last_log = time.time()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024 * 8)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
            now = time.time()
            if now - last_log > 30:
                logger.info("Hashing %s: %.2f GB read", path.name, total / 1e9)
                last_log = now
    return h.hexdigest()


def read_tar_gzip_text_member(tar_path: Path, member_name: str) -> bytes:
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.getmember(member_name)
        fh = tar.extractfile(member)
        if fh is None:
            raise FileNotFoundError(member_name)
        compressed = fh.read()
    return gzip.decompress(compressed)


def read_cellranger_barcodes(matrix_tar: Path) -> list[str]:
    data = read_tar_gzip_text_member(matrix_tar, "barcodes.tsv.gz")
    return [line.decode("utf-8").strip() for line in data.splitlines() if line]


def read_analysis_csv(analysis_tar: Path, member_name: str) -> pd.DataFrame:
    with tarfile.open(analysis_tar, "r:gz") as tar:
        fh = tar.extractfile(member_name)
        if fh is None:
            raise FileNotFoundError(member_name)
        return pd.read_csv(fh)


def load_10x_matrix(matrix_tar: Path) -> tuple[pd.DataFrame, list[str], sparse.csc_matrix]:
    with tarfile.open(matrix_tar, "r:gz") as tar:
        features_fh = tar.extractfile("features.tsv.gz")
        barcodes_fh = tar.extractfile("barcodes.tsv.gz")
        matrix_fh = tar.extractfile("matrix.mtx.gz")
        if features_fh is None or barcodes_fh is None or matrix_fh is None:
            raise FileNotFoundError("10x matrix archive must contain features.tsv.gz, barcodes.tsv.gz, matrix.mtx.gz")

        with gzip.GzipFile(fileobj=features_fh) as gz:
            features = pd.read_csv(
                gz,
                sep="\t",
                header=None,
                names=["feature_id", "gene_name", "feature_type"],
                dtype=str,
            )
        with gzip.GzipFile(fileobj=barcodes_fh) as gz:
            barcodes = [line.decode("utf-8").strip() for line in gz if line.strip()]
        with gzip.GzipFile(fileobj=matrix_fh) as gz:
            matrix = scipy_io.mmread(gz).tocsc()
    return features, barcodes, matrix


def iter_fastq_sequences(path: Path):
    with gzip.open(path, "rb") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().strip().upper()
            fh.readline()
            fh.readline()
            yield seq


def iter_fastq_pairs(r1: Path, r2: Path):
    with gzip.open(r1, "rb") as f1, gzip.open(r2, "rb") as f2:
        while True:
            h1 = f1.readline()
            h2 = f2.readline()
            if not h1 and not h2:
                break
            if not h1 or not h2:
                raise ValueError(f"FASTQ pair length mismatch: {r1} and {r2}")
            seq1 = f1.readline().strip().upper()
            seq2 = f2.readline().strip().upper()
            f1.readline()
            f1.readline()
            f2.readline()
            f2.readline()
            yield seq1, seq2


def probe_fastq(path: Path, n_reads: int, relay_anchor: bytes | None = None) -> dict[str, Any]:
    lengths: Counter[int] = Counter()
    first_sequences: list[str] = []
    anchor_at_zero = 0
    anchor_anywhere = 0
    n = 0
    for seq in iter_fastq_sequences(path):
        n += 1
        lengths[len(seq)] += 1
        if len(first_sequences) < 3:
            first_sequences.append(seq[:80].decode("ascii", errors="replace"))
        if relay_anchor is not None:
            if seq.startswith(relay_anchor):
                anchor_at_zero += 1
            if relay_anchor in seq:
                anchor_anywhere += 1
        if n >= n_reads:
            break
    return {
        "reads_examined": n,
        "length_distribution": {str(k): int(v) for k, v in sorted(lengths.items())},
        "mode_length": int(lengths.most_common(1)[0][0]) if lengths else None,
        "first_sequence_prefixes": first_sequences,
        "relay_anchor_exact_at_position_0": anchor_at_zero if relay_anchor is not None else None,
        "relay_anchor_exact_anywhere": anchor_anywhere if relay_anchor is not None else None,
    }


_RC_TABLE = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")


def reverse_complement(seq: bytes) -> bytes:
    return seq.translate(_RC_TABLE)[::-1].upper()


def build_direct_and_rc_maps(
    config: dict[str, Any],
    peptide_table: pd.DataFrame,
) -> tuple[dict[str, dict[bytes, Any]], dict[str, dict[bytes, Any]]]:
    treatment_direct = {k.encode("ascii"): k for k in config["treatment_barcodes"]}
    treatment_rc = {reverse_complement(k.encode("ascii")): k for k in config["treatment_barcodes"]}
    peptide_direct = {
        str(row["BC"]).encode("ascii"): (str(row["Index"]), str(row["Name"]))
        for _, row in peptide_table.iterrows()
    }
    peptide_rc = {
        reverse_complement(str(row["BC"]).encode("ascii")): (str(row["Index"]), str(row["Name"]))
        for _, row in peptide_table.iterrows()
    }
    return (
        {"direct": treatment_direct, "reverse_complement": treatment_rc},
        {"direct": peptide_direct, "reverse_complement": peptide_rc},
    )


def detect_relay_barcode_orientation(
    config: dict[str, Any],
    root: Path,
    cellbc_set: set[bytes],
    peptide_table: pd.DataFrame,
) -> dict[str, Any]:
    treatment_maps, peptide_maps = build_direct_and_rc_maps(config, peptide_table)
    fastq = config["fastq"]
    anchor = fastq["anchor"].encode("ascii")
    spacer = fastq["spacer"].encode("ascii")
    anchor_len = len(anchor)
    dcbc_end = anchor_len + int(fastq["dcbc_length"])
    treatment_end = dcbc_end + int(fastq["treatment_length"])
    spacer_end = treatment_end + len(spacer)
    peptide_end = spacer_end + int(fastq["peptide_length"])
    n_probe = int(fastq.get("orientation_probe_reads", fastq.get("probe_reads", 10000)))
    cellbc_start = int(fastq["cellbc_start"])
    cellbc_end = cellbc_start + int(fastq["cellbc_length"])

    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {"treatment": [], "peptide": []}
    for pair in config["paths"]["barcode_fastq_pairs"]:
        r1 = resolve(root, pair["r1"])
        r2 = resolve(root, pair["r2"])
        for i, (seq1, seq2) in enumerate(iter_fastq_pairs(r1, r2)):
            if i >= n_probe:
                break
            counts["read_pairs_examined"] += 1
            if len(seq1) < cellbc_end:
                counts["r1_too_short"] += 1
                continue
            if seq1[cellbc_start:cellbc_end] not in cellbc_set:
                counts["cellbc_not_cellranger"] += 1
                continue
            if len(seq2) < peptide_end:
                counts["r2_too_short"] += 1
                continue
            if seq2[:anchor_len] != anchor:
                counts["anchor_mismatch"] += 1
                continue
            if seq2[treatment_end:spacer_end] != spacer:
                counts["spacer_mismatch"] += 1
                continue
            treatment = seq2[dcbc_end:treatment_end]
            peptide = seq2[spacer_end:peptide_end]
            for orientation, mapping in treatment_maps.items():
                if treatment in mapping:
                    counts[f"treatment_{orientation}"] += 1
                    if len(examples["treatment"]) < 5:
                        examples["treatment"].append(
                            {
                                "observed": treatment.decode("ascii"),
                                "orientation": orientation,
                                "canonical": mapping[treatment],
                            }
                        )
            for orientation, mapping in peptide_maps.items():
                if peptide in mapping:
                    counts[f"peptide_{orientation}"] += 1
                    if len(examples["peptide"]) < 5:
                        idx, name = mapping[peptide]
                        examples["peptide"].append(
                            {
                                "observed": peptide.decode("ascii"),
                                "orientation": orientation,
                                "PeptideBC_Index": idx,
                                "PeptideBC_Name": name,
                            }
                        )
            for orientation in ("direct", "reverse_complement"):
                if treatment in treatment_maps[orientation] and peptide in peptide_maps[orientation]:
                    counts[f"both_{orientation}"] += 1

    requested = str(fastq.get("barcode_orientation", "auto"))
    if requested in {"direct", "reverse_complement"}:
        selected = requested
    elif requested == "auto":
        direct = counts.get("both_direct", 0)
        rc = counts.get("both_reverse_complement", 0)
        selected = "reverse_complement" if rc > direct else "direct"
    else:
        raise ValueError(f"Unsupported fastq.barcode_orientation: {requested}")

    return {
        "requested_orientation": requested,
        "selected_orientation": selected,
        "probe_counts": dict(counts),
        "examples": examples,
        "note": "Selected orientation maps observed R2 treatment and peptide segments to canonical config/CSV barcode identities.",
    }


def run_preflight(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    raw_rows = []
    for raw in config["raw_input_files_for_hashing"]:
        path = resolve(root, raw)
        logger.info("Hashing raw input %s", path)
        raw_rows.append(
            {
                "path": str(path),
                "relative_path": os.path.relpath(path, root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path, logger),
            }
        )
    raw_manifest = pd.DataFrame(raw_rows)
    raw_manifest.to_csv(dirs["data"] / "raw_input_manifest.csv", index=False)

    matrix_path = resolve(root, config["paths"]["filtered_feature_matrix"])
    barcodes = read_cellranger_barcodes(matrix_path)
    suffixes = Counter(bc.split("-")[-1] if "-" in bc else "" for bc in barcodes)
    stripped_examples = [bc.split("-")[0] for bc in barcodes[:5]]
    anchor = config["fastq"]["anchor"].encode("ascii")
    probe_reads = int(config["fastq"]["probe_reads"])
    probes: dict[str, Any] = {}
    for pair in config["paths"]["barcode_fastq_pairs"]:
        r1 = resolve(root, pair["r1"])
        r2 = resolve(root, pair["r2"])
        probes[f"{pair['sample']}_R1"] = probe_fastq(r1, probe_reads)
        probes[f"{pair['sample']}_R2"] = probe_fastq(r2, probe_reads, relay_anchor=anchor)
    peptide_table = pd.read_csv(resolve(root, config["paths"]["peptide_barcodes"]), dtype=str)
    orientation_probe = detect_relay_barcode_orientation(
        config,
        root,
        {bc.split("-")[0].encode("ascii") for bc in barcodes},
        peptide_table,
    )

    r1_modes = [probes[f"{pair['sample']}_R1"]["mode_length"] for pair in config["paths"]["barcode_fastq_pairs"]]
    r2_modes = [probes[f"{pair['sample']}_R2"]["mode_length"] for pair in config["paths"]["barcode_fastq_pairs"]]
    min_relay_len = (
        len(config["fastq"]["anchor"])
        + config["fastq"]["dcbc_length"]
        + config["fastq"]["treatment_length"]
        + len(config["fastq"]["spacer"])
        + config["fastq"]["peptide_length"]
    )
    summary = {
        "created_utc": utc_now(),
        "raw_input_manifest": str(dirs["data"] / "raw_input_manifest.csv"),
        "cellranger_filtered_cells": len(barcodes),
        "cellranger_barcode_examples": barcodes[:5],
        "cellranger_barcode_stripped_examples": stripped_examples,
        "cellranger_barcode_suffix_distribution": dict(suffixes),
        "cellranger_barcodes_have_dash_suffix": all("-" in bc for bc in barcodes[: min(100, len(barcodes))]),
        "expected_layout": {
            "R1": "CellBC first 16 nt, UMI next 12 nt",
            "R2": "Relay barcode cassette",
            "relay_minimum_parsed_length": min_relay_len,
        },
        "observed_fastq_probes": probes,
        "relay_barcode_orientation_probe": orientation_probe,
        "layout_confirmed": {
            "r1_mode_lengths_are_28nt": all(length == 28 for length in r1_modes),
            "r2_mode_lengths_cover_relay_cassette": all(length is not None and length >= min_relay_len for length in r2_modes),
            "cellranger_suffix_convention_dash_1_common": suffixes.get("1", 0) > 0,
            "relay_barcode_orientation_has_valid_matches": orientation_probe["probe_counts"].get(
                f"both_{orientation_probe['selected_orientation']}", 0
            )
            > 0,
        },
    }
    write_json(summary, dirs["qc"] / "preflight_summary.json")


def run_doublet_detection(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    analysis_path = resolve(root, config["paths"]["cellranger_analysis"])
    pca = read_analysis_csv(analysis_path, "pca/gene_expression_10_components/projection.csv")
    pc_cols = [c for c in pca.columns if c.startswith("PC-")]
    if not pc_cols:
        raise ValueError("No CellRanger PC columns found in analysis archive.")

    rng = np.random.default_rng(int(config["random_seed"]))
    x = pca[pc_cols].to_numpy(dtype=np.float32)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
    n_obs = x.shape[0]
    sim_ratio = float(config["thresholds"]["doublet_simulation_ratio"])
    n_sim = max(1, int(round(n_obs * sim_ratio)))
    idx1 = rng.integers(0, n_obs, size=n_sim)
    idx2 = rng.integers(0, n_obs, size=n_sim)
    sim = (x[idx1] + x[idx2]) / 2.0
    combined = np.vstack([x, sim]).astype(np.float32, copy=False)
    labels = np.concatenate([np.zeros(n_obs, dtype=np.int8), np.ones(n_sim, dtype=np.int8)])
    k = int(config["thresholds"]["doublet_neighbors"])
    k = max(5, min(k, combined.shape[0] - 1))
    logger.info("Building doublet kNN tree for %d observed and %d simulated cells", n_obs, n_sim)
    tree = cKDTree(combined)
    scores = np.zeros(n_obs, dtype=np.float32)
    batch = 5000
    for start in range(0, n_obs, batch):
        stop = min(start + batch, n_obs)
        _, neighbors = tree.query(x[start:stop], k=k + 1, workers=-1)
        neighbors = neighbors[:, 1:]
        scores[start:stop] = labels[neighbors].mean(axis=1)
        if start and start % 25000 == 0:
            logger.info("Doublet scoring: %d/%d cells", start, n_obs)

    quantile = float(config["thresholds"]["doublet_score_quantile"])
    threshold = float(np.quantile(scores, quantile))
    calls = scores >= threshold
    doublet_scores = pd.DataFrame(
        {
            "CellBC": pca["Barcode"],
            "doublet_score": scores,
            "doublet_call": calls,
        }
    )
    doublet_scores.to_csv(dirs["tables"] / "doublet_scores.csv", index=False)
    doublet_scores.loc[~calls, ["CellBC"]].to_csv(dirs["tables"] / "singlet_barcodes.csv", index=False)
    doublet_scores.loc[calls, ["CellBC"]].to_csv(dirs["tables"] / "doublet_barcodes.csv", index=False)

    summary = {
        "method": "Scrublet-style simulated doublets in CellRanger GEX PCA space with kNN simulated-neighbor fraction.",
        "pca_source": "CellRanger analysis.tar.gz pca/gene_expression_10_components/projection.csv",
        "n_observed_cells": int(n_obs),
        "n_simulated_doublets": int(n_sim),
        "neighbors": int(k),
        "threshold_quantile": quantile,
        "doublet_score_threshold": threshold,
        "n_doublets": int(calls.sum()),
        "n_singlets": int((~calls).sum()),
        "random_seed": int(config["random_seed"]),
    }
    write_json(summary, dirs["qc"] / "doublet_filtering_summary.json")
    plot_histogram(
        scores,
        threshold,
        dirs["qc_plots"],
        "doublet_score_distribution",
        "Doublet score",
        "Cells",
    )


def plot_histogram(values: np.ndarray, threshold: float, out_dir: Path, name: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(2.4, 1.8))
    ax.hist(values, bins=60, color="#8fb3d9", edgecolor="white", linewidth=0.2)
    ax.axvline(threshold, color="#ed8590", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def parse_relay(
    seq: bytes,
    config: dict[str, Any],
    treatment_map: dict[bytes, str],
    peptide_map: dict[bytes, tuple[str, str]],
) -> tuple[str, bytes | None, str | None, tuple[str, str] | None]:
    fastq = config["fastq"]
    anchor = fastq["anchor"].encode("ascii")
    spacer = fastq["spacer"].encode("ascii")
    anchor_len = len(anchor)
    dcbc_start = anchor_len
    dcbc_end = dcbc_start + int(fastq["dcbc_length"])
    treatment_end = dcbc_end + int(fastq["treatment_length"])
    spacer_end = treatment_end + len(spacer)
    peptide_end = spacer_end + int(fastq["peptide_length"])
    if len(seq) < peptide_end:
        return "r2_too_short", None, None, None
    if seq[:anchor_len] != anchor:
        return "anchor_mismatch", None, None, None
    if seq[treatment_end:spacer_end] != spacer:
        return "spacer_mismatch", None, None, None
    dcbc = seq[dcbc_start:dcbc_end]
    treatment = seq[dcbc_end:treatment_end]
    peptide = seq[spacer_end:peptide_end]
    if treatment not in treatment_map:
        return "invalid_treatment", None, None, None
    if peptide not in peptide_map:
        return "invalid_peptide", None, None, None
    return "valid", dcbc, treatment_map[treatment], peptide_map[peptide]


def run_barcode_parse(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    singlets = pd.read_csv(dirs["tables"] / "singlet_barcodes.csv")["CellBC"].astype(str).tolist()
    cellbc_map = {bc.split("-")[0].encode("ascii"): bc for bc in singlets}
    peptides = pd.read_csv(resolve(root, config["paths"]["peptide_barcodes"]), dtype=str)
    all_cellranger_barcodes = read_cellranger_barcodes(resolve(root, config["paths"]["filtered_feature_matrix"]))
    orientation_probe = detect_relay_barcode_orientation(
        config,
        root,
        {bc.split("-")[0].encode("ascii") for bc in all_cellranger_barcodes},
        peptides,
    )
    treatment_maps, peptide_maps = build_direct_and_rc_maps(config, peptides)
    selected_orientation = orientation_probe["selected_orientation"]
    treatment_map = treatment_maps[selected_orientation]
    peptide_map = peptide_maps[selected_orientation]
    umi_start = int(config["fastq"]["umi_start"])
    umi_end = umi_start + int(config["fastq"]["umi_length"])
    cellbc_start = int(config["fastq"]["cellbc_start"])
    cellbc_end = cellbc_start + int(config["fastq"]["cellbc_length"])

    logger.info("Barcode parse pass 1: counting DCBC read support")
    dcbc_counts: Counter[bytes] = Counter()
    qc: dict[str, Any] = {"samples": {}, "created_utc": utc_now()}
    for pair in config["paths"]["barcode_fastq_pairs"]:
        sample = pair["sample"]
        counters: Counter[str] = Counter()
        r1 = resolve(root, pair["r1"])
        r2 = resolve(root, pair["r2"])
        last_log = time.time()
        for seq1, seq2 in iter_fastq_pairs(r1, r2):
            counters["total_read_pairs"] += 1
            if len(seq1) < umi_end:
                counters["r1_too_short"] += 1
                continue
            cellbc_raw = seq1[cellbc_start:cellbc_end]
            if cellbc_raw not in cellbc_map:
                counters["cellbc_not_singlet_cellranger_barcode"] += 1
                continue
            status, dcbc, treatment, peptide = parse_relay(seq2, config, treatment_map, peptide_map)
            if status != "valid":
                counters[status] += 1
                continue
            counters["valid_singlet_mapped_relay_reads_before_dcbc_filter"] += 1
            dcbc_counts[dcbc] += 1
            now = time.time()
            if now - last_log > 30:
                logger.info(
                    "%s pass 1: %.1f M read pairs, %.1f M valid",
                    sample,
                    counters["total_read_pairs"] / 1e6,
                    counters["valid_singlet_mapped_relay_reads_before_dcbc_filter"] / 1e6,
                )
                last_log = now
        qc["samples"][sample] = dict(counters)

    dcbc_rows = [
        {"DCBC": dcbc.decode("ascii"), "total_reads_before_singleton_filter": count, "keep_after_singleton_filter": count > 1}
        for dcbc, count in dcbc_counts.items()
    ]
    if dcbc_rows:
        dcbc_df = pd.DataFrame(dcbc_rows).sort_values("total_reads_before_singleton_filter", ascending=False)
    else:
        dcbc_df = pd.DataFrame(columns=["DCBC", "total_reads_before_singleton_filter", "keep_after_singleton_filter"])
    dcbc_df.to_csv(dirs["data"] / "dcbc_precollapse_read_counts.csv", index=False)
    keep_dcbc = {dcbc for dcbc, count in dcbc_counts.items() if count > 1}
    singleton_reads = sum(count for dcbc, count in dcbc_counts.items() if count <= 1)

    logger.info("Barcode parse pass 2: collapsing UMIs for %d retained DCBCs", len(keep_dcbc))
    db_path = dirs["data"] / "barcode_umi_counts.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;
        CREATE TABLE umi_counts (
            CellBC TEXT NOT NULL,
            PeptideBC_Index TEXT NOT NULL,
            PeptideBC_Name TEXT NOT NULL,
            TreatmentBC TEXT NOT NULL,
            DCBC TEXT NOT NULL,
            UMISeq TEXT NOT NULL,
            Reads INTEGER NOT NULL,
            PRIMARY KEY (CellBC, PeptideBC_Index, PeptideBC_Name, TreatmentBC, DCBC, UMISeq)
        ) WITHOUT ROWID;
        """
    )
    flush_threshold = int(config["fastq"].get("sqlite_flush_unique_keys", 1000000))
    chunk_counter: Counter[tuple[str, str, str, str, str, bytes]] = Counter()
    premerge_rows_flushed = 0

    def flush_umi_chunk() -> int:
        nonlocal chunk_counter, premerge_rows_flushed
        if not chunk_counter:
            return 0
        rows = [
            (key[0], key[1], key[2], key[3], key[4], key[5].decode("ascii"), int(reads))
            for key, reads in chunk_counter.items()
        ]
        cur.executemany(
            """
            INSERT INTO umi_counts
                (CellBC, PeptideBC_Index, PeptideBC_Name, TreatmentBC, DCBC, UMISeq, Reads)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(CellBC, PeptideBC_Index, PeptideBC_Name, TreatmentBC, DCBC, UMISeq)
            DO UPDATE SET Reads = Reads + excluded.Reads
            """,
            rows,
        )
        conn.commit()
        flushed = len(rows)
        premerge_rows_flushed += flushed
        chunk_counter.clear()
        return flushed

    pass2_qc: dict[str, Any] = {"samples": {}}
    for pair in config["paths"]["barcode_fastq_pairs"]:
        sample = pair["sample"]
        counters = Counter()
        r1 = resolve(root, pair["r1"])
        r2 = resolve(root, pair["r2"])
        last_log = time.time()
        for seq1, seq2 in iter_fastq_pairs(r1, r2):
            counters["total_read_pairs"] += 1
            if len(seq1) < umi_end:
                counters["r1_too_short"] += 1
                continue
            cellbc_raw = seq1[cellbc_start:cellbc_end]
            cellbc = cellbc_map.get(cellbc_raw)
            if cellbc is None:
                counters["cellbc_not_singlet_cellranger_barcode"] += 1
                continue
            status, dcbc, treatment, peptide = parse_relay(seq2, config, treatment_map, peptide_map)
            if status != "valid":
                counters[status] += 1
                continue
            if dcbc not in keep_dcbc:
                counters["dcbc_singleton_discarded_reads"] += 1
                continue
            peptide_index, peptide_name = peptide
            umi_seq = seq1[umi_start:umi_end]
            key = (
                cellbc,
                peptide_index,
                peptide_name,
                treatment,
                dcbc.decode("ascii"),
                umi_seq,
            )
            chunk_counter[key] += 1
            counters["final_read_pairs_after_all_filters"] += 1
            if len(chunk_counter) >= flush_threshold:
                flushed = flush_umi_chunk()
                logger.info(
                    "%s pass 2: flushed %.1f M pre-merge UMI keys to SQLite",
                    sample,
                    flushed / 1e6,
                )
            now = time.time()
            if now - last_log > 30:
                logger.info(
                    "%s pass 2: %.1f M read pairs, %.1f M final reads, %.1f M staged UMI keys",
                    sample,
                    counters["total_read_pairs"] / 1e6,
                    counters["final_read_pairs_after_all_filters"] / 1e6,
                    len(chunk_counter) / 1e6,
                )
                last_log = now
        flush_umi_chunk()
        pass2_qc["samples"][sample] = dict(counters)

    raw_count_path = dirs["data"] / "barcode_raw_count_table.csv.gz"
    logger.info("Writing final collapsed barcode count table from SQLite")
    unique_feature_umi_groups = int(cur.execute("SELECT COUNT(*) FROM umi_counts").fetchone()[0])
    feature_rows = 0
    with gzip.open(raw_count_path, "wt", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["CellBC", "PeptideBC_Index", "PeptideBC_Name", "TreatmentBC", "DCBC", "UMI", "Reads"])
        for row in cur.execute(
            """
            SELECT
                CellBC,
                PeptideBC_Index,
                PeptideBC_Name,
                TreatmentBC,
                DCBC,
                COUNT(*) AS UMI,
                SUM(Reads) AS Reads
            FROM umi_counts
            GROUP BY CellBC, PeptideBC_Index, PeptideBC_Name, TreatmentBC, DCBC
            """
        ):
            writer.writerow(row)
            feature_rows += 1
    conn.close()

    total_valid_before = sum(s.get("valid_singlet_mapped_relay_reads_before_dcbc_filter", 0) for s in qc["samples"].values())
    total_final = sum(s.get("final_read_pairs_after_all_filters", 0) for s in pass2_qc["samples"].values())
    qc.update(
        {
            "pass2": pass2_qc,
            "dcbc_total_observed": len(dcbc_counts),
            "dcbc_retained_after_discarding_singleton_read_dcbc": len(keep_dcbc),
            "dcbc_singleton_discarded_reads": int(singleton_reads),
            "valid_singlet_mapped_relay_reads_before_dcbc_filter_total": int(total_valid_before),
            "final_read_pairs_after_all_filters_total": int(total_final),
            "unique_feature_umi_groups": unique_feature_umi_groups,
            "unique_feature_rows": int(feature_rows),
            "umi_sqlite_database": str(db_path),
            "sqlite_flush_unique_keys": flush_threshold,
            "sqlite_premerge_rows_flushed": int(premerge_rows_flushed),
            "raw_count_table": str(raw_count_path),
            "raw_count_table_schema": ["CellBC", "PeptideBC_Index", "PeptideBC_Name", "TreatmentBC", "DCBC", "UMI", "Reads"],
            "relay_barcode_orientation_probe": orientation_probe,
            "relay_barcode_selected_orientation": selected_orientation,
            "treatment_bc_output": "canonical config sequence after orientation mapping",
            "exact_relay_parser": {
                "anchor": config["fastq"]["anchor"],
                "spacer": config["fastq"]["spacer"],
                "invalid anchors were discarded": True,
            },
        }
    )
    write_json(qc, dirs["qc"] / "barcode_parse_qc.json")
    plot_barcode_filter_qc(qc, dirs["qc_plots"])


def plot_barcode_filter_qc(qc: dict[str, Any], out_dir: Path) -> None:
    labels = [
        "mapped+valid\nbefore DCBC",
        "singleton\nDCBC reads",
        "final\nreads",
    ]
    values = [
        qc.get("valid_singlet_mapped_relay_reads_before_dcbc_filter_total", 0),
        qc.get("dcbc_singleton_discarded_reads", 0),
        qc.get("final_read_pairs_after_all_filters_total", 0),
    ]
    fig, ax = plt.subplots(figsize=(2.3, 1.8))
    ax.bar(range(len(values)), values, color=["#8fb3d9", "#cfcfcf", "#93c47d"], linewidth=0)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Read pairs")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(out_dir / "barcode_filter_counts.pdf")
    fig.savefig(out_dir / "barcode_filter_counts.png", dpi=300)
    plt.close(fig)


def build_dcbc_identity(
    config: dict[str, Any],
    dirs: dict[str, Path],
    count_table_path: Path,
    identity_table_name: str,
    components_table_name: str,
    summary_name: str,
) -> None:
    treatment_names = config["treatment_barcodes"]
    agg_umi: Counter[tuple[str, str, str, str]] = Counter()
    agg_reads: Counter[tuple[str, str, str, str]] = Counter()
    for chunk in pd.read_csv(count_table_path, chunksize=500000, dtype={"TreatmentBC": str, "DCBC": str, "PeptideBC_Index": str, "PeptideBC_Name": str}):
        grouped = chunk.groupby(["DCBC", "PeptideBC_Index", "PeptideBC_Name", "TreatmentBC"], as_index=False)[["UMI", "Reads"]].sum()
        for row in grouped.itertuples(index=False):
            key = (row.DCBC, row.PeptideBC_Index, row.PeptideBC_Name, row.TreatmentBC)
            agg_umi[key] += int(row.UMI)
            agg_reads[key] += int(row.Reads)

    by_dcbc: dict[str, list[tuple[tuple[str, str, str, str], int, int]]] = defaultdict(list)
    for key, umi in agg_umi.items():
        by_dcbc[key[0]].append((key, int(umi), int(agg_reads[key])))

    threshold = float(config["thresholds"]["dcbc_identity_dominance_fraction"])
    rows = []
    detail_rows = []
    for dcbc, entries in by_dcbc.items():
        total_umi = sum(e[1] for e in entries)
        total_reads = sum(e[2] for e in entries)
        entries_sorted = sorted(entries, key=lambda x: (x[1], x[2]), reverse=True)
        top_key, top_umi, top_reads = entries_sorted[0]
        frac = top_umi / total_umi if total_umi else 0.0
        status = "assigned" if frac >= threshold else "ambiguous"
        treatment_name = treatment_names.get(top_key[3], "unknown")
        rows.append(
            {
                "DCBC": dcbc,
                "total_umi": total_umi,
                "total_reads": total_reads,
                "dominant_PeptideBC_Index": top_key[1],
                "dominant_PeptideBC_Name": top_key[2],
                "dominant_TreatmentBC": top_key[3],
                "dominant_Treatment": treatment_name,
                "dominant_umi": top_umi,
                "dominant_reads": top_reads,
                "dominance_fraction": frac,
                "dcbc_identity_status": status,
                "AssignedPeptideBC_Index": top_key[1] if status == "assigned" else "",
                "AssignedPeptideBC_Name": top_key[2] if status == "assigned" else "",
                "AssignedTreatmentBC": top_key[3] if status == "assigned" else "",
                "AssignedTreatment": treatment_name if status == "assigned" else "",
            }
        )
        for key, umi, reads in entries_sorted:
            detail_rows.append(
                {
                    "DCBC": dcbc,
                    "PeptideBC_Index": key[1],
                    "PeptideBC_Name": key[2],
                    "TreatmentBC": key[3],
                    "Treatment": treatment_names.get(key[3], "unknown"),
                    "UMI": umi,
                    "Reads": reads,
                    "fraction_of_dcbc_umi": umi / total_umi if total_umi else 0.0,
                }
            )

    identity = pd.DataFrame(rows).sort_values(["dcbc_identity_status", "total_umi"], ascending=[False, False])
    detail = pd.DataFrame(detail_rows)
    identity_path = dirs["data"] / identity_table_name
    components_path = dirs["data"] / components_table_name
    identity.to_csv(identity_path, index=False)
    detail.to_csv(components_path, index=False)
    summary = {
        "source_count_table": str(count_table_path),
        "identity_threshold_dominant_umi_fraction": threshold,
        "n_dcbc": int(len(identity)),
        "n_assigned": int((identity["dcbc_identity_status"] == "assigned").sum()),
        "n_ambiguous": int((identity["dcbc_identity_status"] == "ambiguous").sum()),
        "median_dominance_fraction": float(identity["dominance_fraction"].median()) if len(identity) else None,
        "dcbc_identity_table": str(identity_path),
        "dcbc_identity_components": str(components_path),
    }
    write_json(summary, dirs["qc"] / summary_name)


def run_dcbc_identity_precorrection(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    build_dcbc_identity(
        config,
        dirs,
        dirs["data"] / "barcode_raw_count_table.csv.gz",
        "dcbc_identity_precorrection_table.csv",
        "dcbc_identity_precorrection_components.csv",
        "dcbc_identity_precorrection_summary.json",
    )


def run_dcbc_identity(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    build_dcbc_identity(
        config,
        dirs,
        dirs["data"] / "barcode_corrected_count_table.csv.gz",
        "dcbc_identity_table.csv",
        "dcbc_identity_components.csv",
        "dcbc_identity_summary.json",
    )


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def dcbc_mask_keys(seq: str, max_distance: int) -> list[str]:
    """Return wildcard signatures that identify possible Hamming-distance neighbors."""
    if max_distance > 2:
        raise ValueError("DCBC correction mask index currently supports max_hamming_distance <= 2")
    keys = [seq]
    n = len(seq)
    if max_distance >= 1:
        for i in range(n):
            keys.append(f"{seq[:i]}*{seq[i + 1:]}")
    if max_distance >= 2:
        chars = list(seq)
        for i in range(n):
            for j in range(i + 1, n):
                masked = chars.copy()
                masked[i] = "*"
                masked[j] = "*"
                keys.append("".join(masked))
    return keys


def load_correction_identity_map(path: Path) -> dict[str, tuple[str, str]]:
    df = pd.read_csv(path, dtype=str).fillna("")
    identity: dict[str, tuple[str, str]] = {}
    for row in df.itertuples(index=False):
        peptide = getattr(row, "AssignedPeptideBC_Name")
        treatment = getattr(row, "AssignedTreatment")
        if peptide and treatment:
            identity[getattr(row, "DCBC")] = (peptide, treatment)
    return identity


def make_empty_cluster() -> dict[str, Any]:
    return {
        "umis": set(),
        "reads": 0,
        "feature_reads": Counter(),
        "feature_umis": defaultdict(set),
    }


def add_umi_row_to_cluster(
    clusters: dict[str, dict[str, Any]],
    dcbc: str,
    umi_seq: str,
    reads: int,
    feature: tuple[str, str, str],
) -> None:
    cluster = clusters.setdefault(dcbc, make_empty_cluster())
    cluster["umis"].add(umi_seq)
    cluster["reads"] += int(reads)
    cluster["feature_reads"][feature] += int(reads)
    cluster["feature_umis"][feature].add(umi_seq)


def dominant_cluster_feature(cluster: dict[str, Any]) -> tuple[str, str, str]:
    candidates = []
    for feature, umis in cluster["feature_umis"].items():
        candidates.append((len(umis), cluster["feature_reads"][feature], feature))
    if not candidates:
        return ("", "", "")
    _, _, feature = max(candidates, key=lambda item: (item[0], item[1], item[2][1], item[2][0], item[2][2]))
    return feature


def process_dcbc_correction_cell(
    cellbc: str,
    clusters: dict[str, dict[str, Any]],
    identity_map: dict[str, tuple[str, str]],
    max_hamming: int,
    discard_reads_le: int,
    corrected_writer: csv.writer,
    event_writer: csv.writer,
    qc: Counter[str],
) -> None:
    if not clusters:
        return

    qc["cells_processed"] += 1
    qc["original_cell_dcbc_pairs"] += len(clusters)
    qc["original_cell_dcbc_umis"] += sum(len(cluster["umis"]) for cluster in clusters.values())
    qc["original_cell_dcbc_reads"] += sum(int(cluster["reads"]) for cluster in clusters.values())

    mask_cache: dict[str, list[str]] = {}

    def get_mask_keys(dcbc: str) -> list[str]:
        keys = mask_cache.get(dcbc)
        if keys is None:
            keys = dcbc_mask_keys(dcbc, max_hamming)
            mask_cache[dcbc] = keys
        return keys

    while True:
        collapsed = False
        identity_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for dcbc in clusters:
            identity = identity_map.get(dcbc)
            if identity:
                identity_groups[identity].append(dcbc)

        identity_mask_index: dict[tuple[str, str], dict[str, set[str]]] = {}
        for identity, dcbcs in identity_groups.items():
            mask_index: dict[str, set[str]] = defaultdict(set)
            for dcbc in dcbcs:
                for key in get_mask_keys(dcbc):
                    mask_index[key].add(dcbc)
            identity_mask_index[identity] = mask_index

        candidate_ids = sorted(
            clusters,
            key=lambda dcbc: (len(clusters[dcbc]["umis"]), int(clusters[dcbc]["reads"]), dcbc),
        )
        for candidate_dcbc in candidate_ids:
            if candidate_dcbc not in clusters:
                continue
            candidate_identity = identity_map.get(candidate_dcbc)
            if not candidate_identity:
                continue
            possible_seeds: set[str] = set()
            mask_index = identity_mask_index.get(candidate_identity, {})
            for key in get_mask_keys(candidate_dcbc):
                possible_seeds.update(mask_index.get(key, ()))
            possible_seeds.discard(candidate_dcbc)
            seed_ids = sorted(
                (dcbc for dcbc in possible_seeds if dcbc in clusters),
                key=lambda dcbc: (len(clusters[dcbc]["umis"]), int(clusters[dcbc]["reads"]), dcbc),
                reverse=True,
            )
            for seed_dcbc in seed_ids:
                distance = hamming_distance(candidate_dcbc, seed_dcbc)
                if distance > max_hamming:
                    continue

                candidate = clusters[candidate_dcbc]
                seed = clusters[seed_dcbc]
                candidate_umi_before = len(candidate["umis"])
                seed_umi_before = len(seed["umis"])
                seed_reads_before = int(seed["reads"])
                candidate_reads = int(candidate["reads"])
                shared_umis = len(seed["umis"] & candidate["umis"])
                new_umis = len(candidate["umis"] - seed["umis"])

                seed["umis"].update(candidate["umis"])
                seed["reads"] += candidate_reads
                for feature, reads in candidate["feature_reads"].items():
                    seed["feature_reads"][feature] += int(reads)
                for feature, umis in candidate["feature_umis"].items():
                    seed["feature_umis"][feature].update(umis)

                del clusters[candidate_dcbc]
                qc["collapse_events"] += 1
                qc[f"hamming_{distance}_collapse_events"] += 1
                qc["collapse_reassigned_reads"] += candidate_reads
                qc["collapse_candidate_umi_total"] += candidate_umi_before
                qc["collapse_shared_umi_total"] += shared_umis
                qc["collapse_new_umi_added_total"] += new_umis
                if shared_umis:
                    qc["collapse_events_with_shared_umi"] += 1
                if new_umis:
                    qc["collapse_events_with_new_umi"] += 1

                event_writer.writerow(
                    [
                        qc["collapse_events"],
                        cellbc,
                        candidate_dcbc,
                        seed_dcbc,
                        seed_dcbc,
                        distance,
                        candidate_identity[0],
                        candidate_identity[1],
                        candidate_umi_before,
                        seed_umi_before,
                        candidate_reads,
                        seed_reads_before,
                        shared_umis,
                        new_umis,
                        len(seed["umis"]),
                        int(seed["reads"]),
                    ]
                )
                collapsed = True
                break
            if collapsed:
                break
        if not collapsed:
            break

    qc["after_hamming_cell_dcbc_pairs"] += len(clusters)
    qc["after_hamming_cell_dcbc_umis"] += sum(len(cluster["umis"]) for cluster in clusters.values())
    qc["after_hamming_cell_dcbc_reads"] += sum(int(cluster["reads"]) for cluster in clusters.values())

    for dcbc, cluster in sorted(clusters.items()):
        umi_count = len(cluster["umis"])
        reads = int(cluster["reads"])
        if reads <= discard_reads_le:
            qc["discarded_cell_dcbc_pairs_reads_le_threshold"] += 1
            qc["discarded_cell_dcbc_umis"] += umi_count
            qc["discarded_cell_dcbc_reads"] += reads
            continue
        peptide_index, peptide_name, treatment_bc = dominant_cluster_feature(cluster)
        corrected_writer.writerow([cellbc, peptide_index, peptide_name, treatment_bc, dcbc, umi_count, reads])
        qc["after_read_filter_cell_dcbc_pairs"] += 1
        qc["after_read_filter_cell_dcbc_umis"] += umi_count
        qc["after_read_filter_cell_dcbc_reads"] += reads


def run_dcbc_correction(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    sqlite_path = dirs["data"] / "barcode_umi_counts.sqlite"
    prelim_identity_path = dirs["data"] / "dcbc_identity_precorrection_table.csv"
    corrected_path = dirs["data"] / "barcode_corrected_count_table.csv.gz"
    events_path = dirs["data"] / "dcbc_correction_events.csv.gz"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Missing raw per-UMI SQLite table: {sqlite_path}")
    if not prelim_identity_path.exists():
        raise FileNotFoundError(f"Missing preliminary DCBC identity table: {prelim_identity_path}")

    max_hamming = int(config["thresholds"]["dcbc_correction_max_hamming_distance"])
    discard_reads_le = int(config["thresholds"]["dcbc_correction_discard_cell_dcbc_reads_le"])
    identity_map = load_correction_identity_map(prelim_identity_path)
    qc: Counter[str] = Counter()
    qc["max_hamming_distance"] = max_hamming
    qc["discard_cell_dcbc_reads_le"] = discard_reads_le
    qc["preliminary_assigned_dcbc_identities"] = len(identity_map)

    logger.info("Starting DCBC correction from %s", sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    query = """
        SELECT CellBC, PeptideBC_Index, PeptideBC_Name, TreatmentBC, DCBC, UMISeq, Reads
        FROM umi_counts
        ORDER BY CellBC
    """
    with gzip.open(corrected_path, "wt", newline="") as corrected_fh, gzip.open(events_path, "wt", newline="") as events_fh:
        corrected_writer = csv.writer(corrected_fh)
        corrected_writer.writerow(["CellBC", "PeptideBC_Index", "PeptideBC_Name", "TreatmentBC", "DCBC", "UMI", "Reads"])
        event_writer = csv.writer(events_fh)
        event_writer.writerow(
            [
                "event_id",
                "CellBC",
                "candidate_DCBC",
                "seed_DCBC",
                "corrected_DCBC",
                "hamming_distance",
                "AssignedPeptideBC_Name",
                "AssignedTreatment",
                "candidate_umi_before",
                "seed_umi_before",
                "candidate_reads",
                "seed_reads_before",
                "shared_umi_count",
                "new_umi_added_count",
                "seed_umi_after",
                "seed_reads_after",
            ]
        )

        current_cell = None
        clusters: dict[str, dict[str, Any]] = {}
        last_log = time.time()
        for row in conn.execute(query):
            cellbc, peptide_index, peptide_name, treatment_bc, dcbc, umi_seq, reads = row
            if current_cell is None:
                current_cell = cellbc
            if cellbc != current_cell:
                process_dcbc_correction_cell(
                    current_cell,
                    clusters,
                    identity_map,
                    max_hamming,
                    discard_reads_le,
                    corrected_writer,
                    event_writer,
                    qc,
                )
                now = time.time()
                if now - last_log > 30:
                    logger.info(
                        "DCBC correction: %d cells, %d collapse events, %d kept cell-DCBC pairs",
                        qc["cells_processed"],
                        qc["collapse_events"],
                        qc["after_read_filter_cell_dcbc_pairs"],
                    )
                    last_log = now
                current_cell = cellbc
                clusters = {}
            add_umi_row_to_cluster(
                clusters,
                dcbc,
                umi_seq,
                int(reads),
                (peptide_index, peptide_name, treatment_bc),
            )
        if current_cell is not None:
            process_dcbc_correction_cell(
                current_cell,
                clusters,
                identity_map,
                max_hamming,
                discard_reads_le,
                corrected_writer,
                event_writer,
                qc,
            )
    conn.close()

    summary = {
        "method": "Per-cell recursive bottom-up DCBC Hamming collapse from raw per-UMI SQLite rows, gated by preliminary assigned peptide name and treatment identity.",
        "raw_umi_sqlite": str(sqlite_path),
        "preliminary_identity_table": str(prelim_identity_path),
        "corrected_count_table": str(corrected_path),
        "correction_events": str(events_path),
        "corrected_count_table_schema": ["CellBC", "PeptideBC_Index", "PeptideBC_Name", "TreatmentBC", "DCBC", "UMI", "Reads"],
        "max_hamming_distance": max_hamming,
        "discard_cell_dcbc_reads_le": discard_reads_le,
        "original_cell_dcbc_pairs": int(qc["original_cell_dcbc_pairs"]),
        "after_hamming_cell_dcbc_pairs": int(qc["after_hamming_cell_dcbc_pairs"]),
        "after_read_filter_cell_dcbc_pairs": int(qc["after_read_filter_cell_dcbc_pairs"]),
        "collapse_events": int(qc["collapse_events"]),
        "hamming_1_collapse_events": int(qc["hamming_1_collapse_events"]),
        "hamming_2_collapse_events": int(qc["hamming_2_collapse_events"]),
        "collapse_events_with_shared_umi": int(qc["collapse_events_with_shared_umi"]),
        "collapse_events_with_new_umi": int(qc["collapse_events_with_new_umi"]),
        "collapse_reassigned_reads": int(qc["collapse_reassigned_reads"]),
        "collapse_candidate_umi_total": int(qc["collapse_candidate_umi_total"]),
        "collapse_shared_umi_total": int(qc["collapse_shared_umi_total"]),
        "collapse_new_umi_added_total": int(qc["collapse_new_umi_added_total"]),
        "discarded_cell_dcbc_pairs_reads_le_threshold": int(qc["discarded_cell_dcbc_pairs_reads_le_threshold"]),
        "discarded_cell_dcbc_umis": int(qc["discarded_cell_dcbc_umis"]),
        "discarded_cell_dcbc_reads": int(qc["discarded_cell_dcbc_reads"]),
        "original_cell_dcbc_umis": int(qc["original_cell_dcbc_umis"]),
        "original_cell_dcbc_reads": int(qc["original_cell_dcbc_reads"]),
        "after_hamming_cell_dcbc_umis": int(qc["after_hamming_cell_dcbc_umis"]),
        "after_hamming_cell_dcbc_reads": int(qc["after_hamming_cell_dcbc_reads"]),
        "after_read_filter_cell_dcbc_umis": int(qc["after_read_filter_cell_dcbc_umis"]),
        "after_read_filter_cell_dcbc_reads": int(qc["after_read_filter_cell_dcbc_reads"]),
        "cells_processed": int(qc["cells_processed"]),
        "preliminary_assigned_dcbc_identities": int(qc["preliminary_assigned_dcbc_identities"]),
    }
    write_json(summary, dirs["qc"] / "dcbc_correction_summary.json")


def downstream_barcode_count_path(dirs: dict[str, Path]) -> Path:
    return dirs["data"] / "barcode_read_support_filtered_count_table.csv.gz"


def fit_huber_line(x: np.ndarray, y: np.ndarray, tuning: float, max_iter: int = 100, tol: float = 1e-10) -> dict[str, float]:
    if len(x) < 2:
        intercept = float(np.median(y)) if len(y) else 0.0
        return {"intercept": intercept, "slope": 0.0, "iterations": 0.0}

    design = np.column_stack([np.ones(len(x), dtype=float), x.astype(float)])
    beta = np.linalg.lstsq(design, y.astype(float), rcond=None)[0]
    iterations = 0
    for i in range(max_iter):
        fitted = design @ beta
        residual = y - fitted
        residual_median = float(np.median(residual))
        mad_sigma = float(1.4826 * np.median(np.abs(residual - residual_median)))
        if mad_sigma <= 1e-12:
            break
        scaled = np.abs((residual - residual_median) / mad_sigma)
        weights = np.ones_like(scaled, dtype=float)
        high = scaled > tuning
        weights[high] = tuning / np.maximum(scaled[high], 1e-12)
        sqrt_w = np.sqrt(weights)
        beta_next = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
        iterations = i + 1
        if np.max(np.abs(beta_next - beta)) < tol:
            beta = beta_next
            break
        beta = beta_next

    return {"intercept": float(beta[0]), "slope": float(beta[1]), "iterations": float(iterations)}


def summarize_flagged_conditions(flagged: pd.DataFrame) -> list[dict[str, Any]]:
    if flagged.empty:
        return []
    grouped = (
        flagged.groupby(["AssignedPeptideBC_Name", "AssignedTreatment"], dropna=False)
        .agg(
            flagged_edges=("CellBC", "size"),
            flagged_unique_cells=("CellBC", "nunique"),
            flagged_unique_dcbc=("DCBC", "nunique"),
            flagged_umi=("UMI", "sum"),
            flagged_reads=("Reads", "sum"),
            median_reads_per_umi=("Reads_per_UMI", "median"),
        )
        .reset_index()
        .sort_values(["AssignedTreatment", "AssignedPeptideBC_Name"])
    )
    records: list[dict[str, Any]] = []
    for row in grouped.itertuples(index=False):
        records.append(
            {
                "AssignedPeptideBC_Name": "" if pd.isna(row.AssignedPeptideBC_Name) else str(row.AssignedPeptideBC_Name),
                "AssignedTreatment": "" if pd.isna(row.AssignedTreatment) else str(row.AssignedTreatment),
                "flagged_edges": int(row.flagged_edges),
                "flagged_unique_cells": int(row.flagged_unique_cells),
                "flagged_unique_dcbc": int(row.flagged_unique_dcbc),
                "flagged_umi": int(row.flagged_umi),
                "flagged_reads": int(row.flagged_reads),
                "median_reads_per_umi": float(row.median_reads_per_umi),
            }
        )
    return records


def plot_read_support_filter(edges: pd.DataFrame, class_summaries: dict[str, dict[str, Any]], dirs: dict[str, Path]) -> None:
    class_labels = {"t_cell": "T cells", "dendritic_cell": "Dendritic cells"}
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.35), sharex=True, sharey=True)
    for ax, cell_class in zip(axes, ["t_cell", "dendritic_cell"]):
        sub = edges[(edges["cell_class"] == cell_class) & (edges["is_assigned_dcbc"])].copy()
        flagged = sub[sub["low_read_support"]]
        kept = sub[~sub["low_read_support"]]
        if not kept.empty:
            ax.scatter(
                kept["log10_UMI"],
                kept["log10_Reads"],
                s=3,
                color="#b7b7b7",
                alpha=0.35,
                linewidths=0,
                rasterized=True,
            )
        if not flagged.empty:
            ax.scatter(
                flagged["log10_UMI"],
                flagged["log10_Reads"],
                s=10,
                color="#ed8590",
                edgecolors="#7a2f36",
                linewidths=0.25,
                alpha=0.9,
                rasterized=True,
            )
        summary = class_summaries.get(cell_class, {})
        if not sub.empty and "fit_intercept" in summary and "fit_slope" in summary:
            x_min = float(sub["log10_UMI"].min())
            x_max = float(sub["log10_UMI"].max())
            xs = np.linspace(x_min, x_max, 100)
            ys = float(summary["fit_intercept"]) + float(summary["fit_slope"]) * xs
            ax.plot(xs, ys, color="#4a4a4a", linewidth=0.8)
            threshold = summary.get("residual_threshold")
            if threshold is not None:
                ax.plot(xs, ys + float(threshold), color="#ed8590", linewidth=0.7, linestyle=":")
        ax.set_title(f"{class_labels[cell_class]} ({len(flagged)} flagged)", fontsize=7)
        ax.set_xlabel("log10(UMI)")
        ax.grid(True, color="#e6e6e6", linewidth=0.35)
    axes[0].set_ylabel("log10(Reads)")
    fig.tight_layout()
    save_dual(fig, "read_support_tcell_vs_dendritic_cell", dirs)


def run_read_support_filter(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    corrected_path = dirs["data"] / "barcode_corrected_count_table.csv.gz"
    filtered_path = downstream_barcode_count_path(dirs)
    edge_path = dirs["data"] / "read_support_edge_table.csv.gz"
    low_edges_path = dirs["data"] / "read_support_low_support_edges.csv"
    cell_meta_path = dirs["data"] / "cell_metadata.csv"
    identity_path = dirs["data"] / "dcbc_identity_table.csv"

    if not corrected_path.exists():
        raise FileNotFoundError(f"Missing corrected count table: {corrected_path}")
    if not cell_meta_path.exists():
        raise FileNotFoundError(f"Missing cell metadata: {cell_meta_path}")
    if not identity_path.exists():
        raise FileNotFoundError(f"Missing final DCBC identity table: {identity_path}")

    thresholds = config["thresholds"]
    enabled = bool(thresholds.get("read_support_filter_enabled", True))
    min_umi = int(thresholds.get("read_support_min_umi_for_outlier", 10))
    mad_multiplier = float(thresholds.get("read_support_mad_multiplier", 4.5))
    huber_tuning = float(thresholds.get("read_support_huber_tuning", 1.345))
    filter_classes = [str(x) for x in thresholds.get("read_support_filter_cell_classes", ["t_cell", "dendritic_cell"])]

    logger.info("Building read-support edge table from %s", corrected_path)
    count_df = pd.read_csv(corrected_path, dtype={"CellBC": str, "DCBC": str, "PeptideBC_Index": str, "PeptideBC_Name": str, "TreatmentBC": str})
    edges = (
        count_df.groupby(["CellBC", "DCBC"], as_index=False)[["UMI", "Reads"]]
        .sum()
        .astype({"UMI": int, "Reads": int})
    )
    cell_meta = pd.read_csv(cell_meta_path, dtype={"CellBC": str})[["CellBC", "cell_class"]]
    identity_cols = [
        "DCBC",
        "dcbc_identity_status",
        "AssignedPeptideBC_Index",
        "AssignedPeptideBC_Name",
        "AssignedTreatmentBC",
        "AssignedTreatment",
    ]
    identity = pd.read_csv(identity_path, dtype=str).fillna("")
    identity = identity[identity_cols]
    edges = edges.merge(cell_meta, on="CellBC", how="left").merge(identity, on="DCBC", how="left")
    for col in identity_cols[1:]:
        edges[col] = edges[col].fillna("")
    edges["cell_class"] = edges["cell_class"].fillna("")
    edges["is_assigned_dcbc"] = (
        (edges["dcbc_identity_status"] == "assigned")
        & edges["AssignedPeptideBC_Name"].ne("")
        & edges["AssignedTreatment"].ne("")
    )
    edges["Reads_per_UMI"] = edges["Reads"] / edges["UMI"].replace(0, np.nan)
    edges["log10_UMI"] = np.log10(edges["UMI"].clip(lower=1))
    edges["log10_Reads"] = np.log10(edges["Reads"].clip(lower=1))
    edges["fit_intercept"] = np.nan
    edges["fit_slope"] = np.nan
    edges["predicted_log10_Reads"] = np.nan
    edges["read_support_residual"] = np.nan
    edges["residual_median"] = np.nan
    edges["mad_sigma"] = np.nan
    edges["residual_threshold"] = np.nan
    edges["low_read_support"] = False

    class_summaries: dict[str, dict[str, Any]] = {}
    flagged_masks = []
    for cell_class in filter_classes:
        class_mask = edges["cell_class"] == cell_class
        assigned_mask = class_mask & edges["is_assigned_dcbc"]
        fit_df = edges.loc[assigned_mask].copy()
        class_total_umi = int(edges.loc[class_mask, "UMI"].sum())
        class_total_reads = int(edges.loc[class_mask, "Reads"].sum())
        if fit_df.empty or not enabled:
            summary = {
                "enabled": enabled,
                "fit_intercept": None,
                "fit_slope": None,
                "fit_iterations": 0,
                "residual_median": None,
                "mad_sigma": None,
                "residual_threshold": None,
                "edge_rows_before": int(class_mask.sum()),
                "assigned_edge_rows_before": int(assigned_mask.sum()),
                "edge_rows_after": int(class_mask.sum()),
                "assigned_edge_rows_after": int(assigned_mask.sum()),
                "flagged_edges": 0,
                "flagged_unique_cells": 0,
                "flagged_unique_dcbc": 0,
                "flagged_umi": 0,
                "flagged_reads": 0,
                "retained_umi": class_total_umi,
                "retained_reads": class_total_reads,
                "median_reads_per_umi_all_edges": float(edges.loc[class_mask, "Reads_per_UMI"].median()) if class_mask.any() else None,
                "median_reads_per_umi_flagged_edges": None,
                "condition_level_flagged_edges": [],
            }
            class_summaries[cell_class] = summary
            continue

        fit = fit_huber_line(fit_df["log10_UMI"].to_numpy(dtype=float), fit_df["log10_Reads"].to_numpy(dtype=float), huber_tuning)
        assigned_pred = fit["intercept"] + fit["slope"] * edges.loc[assigned_mask, "log10_UMI"].to_numpy(dtype=float)
        assigned_residual = edges.loc[assigned_mask, "log10_Reads"].to_numpy(dtype=float) - assigned_pred
        residual_median = float(np.median(assigned_residual))
        mad_sigma = float(1.4826 * np.median(np.abs(assigned_residual - residual_median)))
        residual_threshold = float(residual_median - mad_multiplier * mad_sigma)

        class_pred = fit["intercept"] + fit["slope"] * edges.loc[class_mask, "log10_UMI"].to_numpy(dtype=float)
        edges.loc[class_mask, "fit_intercept"] = fit["intercept"]
        edges.loc[class_mask, "fit_slope"] = fit["slope"]
        edges.loc[class_mask, "predicted_log10_Reads"] = class_pred
        edges.loc[class_mask, "read_support_residual"] = edges.loc[class_mask, "log10_Reads"].to_numpy(dtype=float) - class_pred
        edges.loc[class_mask, "residual_median"] = residual_median
        edges.loc[class_mask, "mad_sigma"] = mad_sigma
        edges.loc[class_mask, "residual_threshold"] = residual_threshold

        flag_mask = assigned_mask & (edges["UMI"] >= min_umi) & (edges["read_support_residual"] < residual_threshold)
        edges.loc[flag_mask, "low_read_support"] = True
        flagged_masks.append(flag_mask)
        flagged = edges.loc[flag_mask].copy()
        retained = edges.loc[class_mask & ~flag_mask]
        summary = {
            "enabled": enabled,
            "fit_intercept": float(fit["intercept"]),
            "fit_slope": float(fit["slope"]),
            "fit_iterations": int(fit["iterations"]),
            "residual_median": residual_median,
            "mad_sigma": mad_sigma,
            "residual_threshold": residual_threshold,
            "edge_rows_before": int(class_mask.sum()),
            "assigned_edge_rows_before": int(assigned_mask.sum()),
            "edge_rows_after": int(len(retained)),
            "assigned_edge_rows_after": int((assigned_mask & ~flag_mask).sum()),
            "flagged_edges": int(flag_mask.sum()),
            "flagged_unique_cells": int(flagged["CellBC"].nunique()),
            "flagged_unique_dcbc": int(flagged["DCBC"].nunique()),
            "flagged_umi": int(flagged["UMI"].sum()),
            "flagged_reads": int(flagged["Reads"].sum()),
            "retained_umi": int(retained["UMI"].sum()),
            "retained_reads": int(retained["Reads"].sum()),
            "median_reads_per_umi_all_edges": float(edges.loc[class_mask, "Reads_per_UMI"].median()) if class_mask.any() else None,
            "median_reads_per_umi_flagged_edges": float(flagged["Reads_per_UMI"].median()) if not flagged.empty else None,
            "condition_level_flagged_edges": summarize_flagged_conditions(flagged),
        }
        class_summaries[cell_class] = summary

    if flagged_masks:
        combined_flag_mask = np.logical_or.reduce([mask.to_numpy(dtype=bool) for mask in flagged_masks])
    else:
        combined_flag_mask = np.zeros(len(edges), dtype=bool)
    flagged_edges = edges.loc[combined_flag_mask].copy()
    flagged_edge_set = set(zip(flagged_edges["CellBC"].astype(str), flagged_edges["DCBC"].astype(str)))

    low_edges_columns = [
        "CellBC",
        "DCBC",
        "cell_class",
        "UMI",
        "Reads",
        "Reads_per_UMI",
        "AssignedPeptideBC_Index",
        "AssignedPeptideBC_Name",
        "AssignedTreatmentBC",
        "AssignedTreatment",
        "log10_UMI",
        "log10_Reads",
        "predicted_log10_Reads",
        "read_support_residual",
        "residual_threshold",
    ]
    flagged_edges[low_edges_columns].to_csv(low_edges_path, index=False)
    edges.to_csv(edge_path, index=False, compression="gzip")

    logger.info("Writing read-support filtered count table with %d flagged edges removed", len(flagged_edge_set))
    written_rows = 0
    written_umi = 0
    written_reads = 0
    with gzip.open(filtered_path, "wt", newline="") as fh:
        writer = csv.writer(fh)
        clean_columns = ["CellBC", "PeptideBC_Index", "PeptideBC_Name", "TreatmentBC", "DCBC", "UMI", "Reads"]
        writer.writerow(clean_columns)
        for chunk in pd.read_csv(corrected_path, chunksize=500000, dtype={"CellBC": str, "DCBC": str, "PeptideBC_Index": str, "PeptideBC_Name": str, "TreatmentBC": str}):
            keep_mask = [(cell, dcbc) not in flagged_edge_set for cell, dcbc in zip(chunk["CellBC"].astype(str), chunk["DCBC"].astype(str))]
            kept = chunk.loc[keep_mask, clean_columns]
            if kept.empty:
                continue
            writer.writerows(kept.itertuples(index=False, name=None))
            written_rows += int(len(kept))
            written_umi += int(kept["UMI"].sum())
            written_reads += int(kept["Reads"].sum())

    plot_read_support_filter(edges, class_summaries, dirs)

    summary = {
        "enabled": enabled,
        "source_count_table": str(corrected_path),
        "filtered_count_table": str(filtered_path),
        "filtered_count_table_schema": ["CellBC", "PeptideBC_Index", "PeptideBC_Name", "TreatmentBC", "DCBC", "UMI", "Reads"],
        "edge_table": str(edge_path),
        "low_support_edges": str(low_edges_path),
        "plot_pdf": str(dirs["fig_pdf"] / "read_support_tcell_vs_dendritic_cell.pdf"),
        "plot_png": str(dirs["fig_png"] / "read_support_tcell_vs_dendritic_cell.png"),
        "read_support_min_umi_for_outlier": min_umi,
        "read_support_mad_multiplier": mad_multiplier,
        "read_support_huber_tuning": huber_tuning,
        "read_support_filter_cell_classes": filter_classes,
        "input_count_rows": int(len(count_df)),
        "input_count_umi": int(count_df["UMI"].sum()),
        "input_count_reads": int(count_df["Reads"].sum()),
        "filtered_count_rows": int(written_rows),
        "filtered_count_umi": int(written_umi),
        "filtered_count_reads": int(written_reads),
        "flagged_edges_total": int(len(flagged_edges)),
        "flagged_umi_total": int(flagged_edges["UMI"].sum()) if not flagged_edges.empty else 0,
        "flagged_reads_total": int(flagged_edges["Reads"].sum()) if not flagged_edges.empty else 0,
        "classes": class_summaries,
    }
    write_json(summary, dirs["qc"] / "read_support_filter_summary.json")


def build_gene_index(features: pd.DataFrame) -> dict[str, list[int]]:
    gene_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in features.iterrows():
        if str(row.get("feature_type", "")) == "Gene Expression":
            gene_to_indices[str(row["gene_name"])].append(int(i))
    return gene_to_indices


def marker_score(
    matrix: sparse.csc_matrix,
    gene_to_indices: dict[str, list[int]],
    marker_genes: list[str],
    cell_indices: np.ndarray,
    library_size: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str]]:
    rows = []
    present = []
    missing = []
    for gene in marker_genes:
        if gene in gene_to_indices:
            rows.append(gene_to_indices[gene][0])
            present.append(gene)
        else:
            missing.append(gene)
    if not rows:
        return np.zeros(len(cell_indices), dtype=np.float32), present, missing
    sub = matrix[rows, :][:, cell_indices].toarray().astype(np.float32)
    denom = np.maximum(library_size[cell_indices].astype(np.float32), 1.0)
    norm = sub / denom[None, :] * 10000.0
    score = np.log1p(norm).mean(axis=0)
    return score, present, missing


def run_cell_classification(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    matrix_path = resolve(root, config["paths"]["filtered_feature_matrix"])
    logger.info("Loading filtered GEX matrix for cell classification")
    features, barcodes, matrix = load_10x_matrix(matrix_path)
    gene_to_indices = build_gene_index(features)
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    singlets = pd.read_csv(dirs["tables"] / "singlet_barcodes.csv")["CellBC"].astype(str).tolist()
    singlets = [bc for bc in singlets if bc in barcode_to_idx]
    cell_indices = np.array([barcode_to_idx[bc] for bc in singlets], dtype=int)
    library_size = np.asarray(matrix.sum(axis=0)).ravel()
    n_genes = np.diff(matrix.indptr)

    dc_score, dc_present, dc_missing = marker_score(
        matrix,
        gene_to_indices,
        config["marker_genes"]["dendritic_cell"],
        cell_indices,
        library_size,
    )
    t_score, t_present, t_missing = marker_score(
        matrix,
        gene_to_indices,
        config["marker_genes"]["t_cell"],
        cell_indices,
        library_size,
    )
    min_score = float(config["thresholds"]["marker_min_score"])
    margin = float(config["thresholds"]["marker_score_margin"])
    labels = np.full(len(singlets), "ambiguous_or_other", dtype=object)
    labels[(dc_score >= min_score) & (dc_score > t_score + margin)] = "dendritic_cell"
    labels[(t_score >= min_score) & (t_score > dc_score + margin)] = "t_cell"

    doublet_scores = pd.read_csv(dirs["tables"] / "doublet_scores.csv")
    doublet_map = dict(zip(doublet_scores["CellBC"].astype(str), doublet_scores["doublet_score"]))
    rows = []
    for pos, bc in enumerate(singlets):
        idx = cell_indices[pos]
        rows.append(
            {
                "CellBC": bc,
                "n_counts": int(library_size[idx]),
                "n_genes": int(n_genes[idx]),
                "doublet_score": float(doublet_map.get(bc, np.nan)),
                "singlet": True,
                "dc_marker_score": float(dc_score[pos]),
                "t_marker_score": float(t_score[pos]),
                "cell_class": labels[pos],
            }
        )
    metadata = pd.DataFrame(rows)
    metadata.to_csv(dirs["data"] / "cell_metadata.csv", index=False)
    summary = {
        "n_singlets_classified": int(len(metadata)),
        "classification_counts": metadata["cell_class"].value_counts().to_dict(),
        "thresholds": {"marker_min_score": min_score, "marker_score_margin": margin},
        "dc_markers_present": dc_present,
        "dc_markers_missing": dc_missing,
        "t_markers_present": t_present,
        "t_markers_missing": t_missing,
        "cell_metadata": str(dirs["data"] / "cell_metadata.csv"),
    }
    write_json(summary, dirs["qc"] / "classification_qc.json")
    plot_classification_qc(metadata, dirs["qc_plots"])


def plot_classification_qc(metadata: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.2, 2.0))
    color_map = {"dendritic_cell": "#8fb3d9", "t_cell": "#f6b26b", "ambiguous_or_other": "#cfcfcf"}
    for label, group in metadata.groupby("cell_class"):
        ax.scatter(
            group["dc_marker_score"],
            group["t_marker_score"],
            s=2,
            alpha=0.5,
            linewidths=0,
            color=color_map.get(label, "#b7b7b7"),
            label=label.replace("_", " "),
        )
    ax.set_xlabel("DC marker score")
    ax.set_ylabel("T marker score")
    ax.legend(markerscale=3, fontsize=5)
    fig.tight_layout()
    fig.savefig(out_dir / "cell_classification_marker_scores.pdf")
    fig.savefig(out_dir / "cell_classification_marker_scores.png", dpi=300)
    plt.close(fig)


def aggregate_cell_dcbc(raw_path: Path, cells: set[str]) -> dict[str, Counter[str]]:
    by_cell: dict[str, Counter[str]] = defaultdict(Counter)
    for chunk in pd.read_csv(raw_path, chunksize=500000, dtype={"CellBC": str, "DCBC": str}):
        chunk = chunk[chunk["CellBC"].isin(cells)]
        if chunk.empty:
            continue
        grouped = chunk.groupby(["CellBC", "DCBC"], as_index=False)["UMI"].sum()
        for row in grouped.itertuples(index=False):
            by_cell[row.CellBC][row.DCBC] += int(row.UMI)
    return by_cell


def load_dcbc_identity_map(path: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(path, dtype=str)
    assigned = df[df["dcbc_identity_status"] == "assigned"].copy()
    return {row["DCBC"]: row.to_dict() for _, row in assigned.iterrows()}


def run_t_cell_metadata(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    cell_meta = pd.read_csv(dirs["data"] / "cell_metadata.csv", dtype={"CellBC": str})
    t_cells = cell_meta.loc[cell_meta["cell_class"] == "t_cell", "CellBC"].astype(str).tolist()
    t_set = set(t_cells)
    filtered_count_path = downstream_barcode_count_path(dirs)
    by_cell = aggregate_cell_dcbc(filtered_count_path, t_set)
    identity_map = load_dcbc_identity_map(dirs["data"] / "dcbc_identity_table.csv")
    vdj = pd.read_csv(resolve(root, config["paths"]["vdj_filtered_contigs"]), dtype=str)
    productive = vdj.get("productive", pd.Series("", index=vdj.index)).astype(str).str.lower() == "true"
    high_conf = vdj.get("high_confidence", pd.Series("", index=vdj.index)).astype(str).str.lower() == "true"
    eligible = productive | high_conf
    trb = config["oti_tcr"]["trb_cdr3_aa"]
    tra = config["oti_tcr"]["tra_cdr3_aa"]
    oti_mask = eligible & (
        ((vdj["chain"] == "TRB") & (vdj["cdr3"] == trb))
        | ((vdj["chain"] == "TRA") & (vdj["cdr3"] == tra))
    )
    oti_barcodes = set(vdj.loc[oti_mask, "barcode"].astype(str))
    clonotype_map: dict[str, str] = {}
    if "raw_clonotype_id" in vdj.columns:
        for bc, sub in vdj.dropna(subset=["raw_clonotype_id"]).groupby("barcode"):
            modes = sub["raw_clonotype_id"].mode()
            clonotype_map[str(bc)] = str(modes.iloc[0]) if len(modes) else ""

    rows = []
    threshold = float(config["thresholds"].get("t_cell_peptide_treatment_fraction", 0.85))
    for bc in t_cells:
        dcbc_counter = by_cell.get(bc, Counter())
        total = int(sum(dcbc_counter.values()))
        top_dcbc = ""
        top_umi = 0
        top_frac = 0.0
        top_peptide_index = ""
        top_peptide_name = ""
        top_peptide_umi = 0
        top_peptide_frac = 0.0
        top_treatment_bc = ""
        top_treatment = ""
        top_treatment_umi = 0
        top_treatment_frac = 0.0
        assigned_peptide_index = ""
        assigned_peptide_name = ""
        assigned_treatment_bc = ""
        assigned_treatment = ""
        state = "no_interaction"
        if total > 0:
            top_dcbc, top_umi = dcbc_counter.most_common(1)[0]
            top_frac = top_umi / total

            peptide_counts: Counter[str] = Counter()
            peptide_index_counts: dict[str, Counter[str]] = defaultdict(Counter)
            treatment_counts: Counter[tuple[str, str]] = Counter()
            assigned_dcbc_umi = 0
            for dcbc, umi in dcbc_counter.items():
                identity = identity_map.get(dcbc)
                if not identity:
                    continue
                assigned_dcbc_umi += int(umi)
                peptide_key = (
                    identity.get("AssignedPeptideBC_Index", ""),
                    identity.get("AssignedPeptideBC_Name", ""),
                )
                treatment_key = (
                    identity.get("AssignedTreatmentBC", ""),
                    identity.get("AssignedTreatment", ""),
                )
                if peptide_key[0] and peptide_key[1]:
                    peptide_counts[peptide_key[1]] += int(umi)
                    peptide_index_counts[peptide_key[1]][peptide_key[0]] += int(umi)
                if treatment_key[0] and treatment_key[1]:
                    treatment_counts[treatment_key] += int(umi)

            if peptide_counts:
                top_peptide_name, top_peptide_umi = max(
                    peptide_counts.items(),
                    key=lambda item: (item[1], item[0]),
                )
                top_peptide_index, _ = max(
                    peptide_index_counts[top_peptide_name].items(),
                    key=lambda item: (item[1], item[0]),
                )
                top_peptide_frac = top_peptide_umi / total
                if top_peptide_frac >= threshold:
                    assigned_peptide_index = top_peptide_index
                    assigned_peptide_name = top_peptide_name

            if treatment_counts:
                (top_treatment_bc, top_treatment), top_treatment_umi = max(
                    treatment_counts.items(),
                    key=lambda item: (item[1], item[0][1], item[0][0]),
                )
                top_treatment_frac = top_treatment_umi / total
                if top_treatment_frac >= threshold:
                    assigned_treatment_bc = top_treatment_bc
                    assigned_treatment = top_treatment

            if assigned_treatment:
                state = "single_interaction"
            else:
                state = "multi_interaction"
        else:
            assigned_dcbc_umi = 0
        rows.append(
            {
                "CellBC": bc,
                "t_cell_type": "OTI" if bc in oti_barcodes else "C57BL6",
                "clonotype_id": clonotype_map.get(bc, ""),
                "total_barcode_umi": total,
                "assigned_dcbc_umi": int(assigned_dcbc_umi),
                "top_DCBC": top_dcbc,
                "top_DCBC_UMI": int(top_umi),
                "top_DCBC_fraction": float(top_frac),
                "top_PeptideBC_Index": top_peptide_index,
                "top_PeptideBC_Name": top_peptide_name,
                "top_Peptide_UMI": int(top_peptide_umi),
                "top_Peptide_fraction": float(top_peptide_frac),
                "top_TreatmentBC": top_treatment_bc,
                "top_Treatment": top_treatment,
                "top_Treatment_UMI": int(top_treatment_umi),
                "top_Treatment_fraction": float(top_treatment_frac),
                "interaction_state": state,
                "AssignedPeptideBC_Index": assigned_peptide_index,
                "AssignedPeptideBC_Name": assigned_peptide_name,
                "AssignedTreatmentBC": assigned_treatment_bc,
                "AssignedTreatment": assigned_treatment,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["data"] / "t_cell_metadata.csv", index=False)
    summary = {
        "n_t_cells": int(len(out)),
        "t_cell_type_counts": out["t_cell_type"].value_counts().to_dict(),
        "interaction_state_counts": out["interaction_state"].value_counts().to_dict(),
        "assignment_rule": "Peptide names and treatments are assigned independently from summed UMIs over assigned DCBC identities. PeptideBC_Index reports the dominant peptide barcode index within the assigned peptide name. Fractions use total barcode UMI as denominator. Barcode-positive cells without assigned treatment are multi_interaction; barcode-zero cells remain no_interaction.",
        "peptide_assignment_fraction_threshold": threshold,
        "treatment_assignment_fraction_threshold": threshold,
        "n_with_assigned_peptide": int(out["AssignedPeptideBC_Name"].fillna("").ne("").sum()),
        "n_with_assigned_treatment": int(out["AssignedTreatment"].fillna("").ne("").sum()),
        "barcode_count_table": str(filtered_count_path),
        "total_barcode_umi_definition": "Corrected DCBC UMI after read-support low-edge filtering.",
        "oti_call": {
            "TRB_CDR3_aa": trb,
            "TRA_CDR3_aa": tra,
            "eligible_contig_rule": "productive OR high_confidence",
        },
        "t_cell_metadata": str(dirs["data"] / "t_cell_metadata.csv"),
    }
    write_json(summary, dirs["qc"] / "t_cell_metadata_summary.json")


def run_dendritic_cell_metadata(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    cell_meta = pd.read_csv(dirs["data"] / "cell_metadata.csv", dtype={"CellBC": str})
    dc_cells = cell_meta.loc[cell_meta["cell_class"] == "dendritic_cell", "CellBC"].astype(str).tolist()
    dc_set = set(dc_cells)
    filtered_count_path = downstream_barcode_count_path(dirs)
    by_cell = aggregate_cell_dcbc(filtered_count_path, dc_set)
    identity_map = load_dcbc_identity_map(dirs["data"] / "dcbc_identity_table.csv")
    threshold = float(config["thresholds"]["dc_single_dcbc_fraction"])
    rows = []
    for bc in dc_cells:
        dcbc_counter = by_cell.get(bc, Counter())
        total = int(sum(dcbc_counter.values()))
        top_dcbc = ""
        top_umi = 0
        top_frac = 0.0
        assigned = {}
        if total > 0:
            top_dcbc, top_umi = dcbc_counter.most_common(1)[0]
            top_frac = top_umi / total
            if top_frac > threshold:
                assigned = identity_map.get(top_dcbc, {})
        rows.append(
            {
                "CellBC": bc,
                "total_barcode_umi": total,
                "top_DCBC": top_dcbc,
                "top_DCBC_UMI": int(top_umi),
                "top_DCBC_fraction": float(top_frac),
                "AssignedPeptideBC_Index": assigned.get("AssignedPeptideBC_Index", ""),
                "AssignedPeptideBC_Name": assigned.get("AssignedPeptideBC_Name", ""),
                "AssignedTreatmentBC": assigned.get("AssignedTreatmentBC", ""),
                "AssignedTreatment": assigned.get("AssignedTreatment", ""),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["data"] / "dendritic_cell_metadata.csv", index=False)
    summary = {
        "n_dendritic_cells": int(len(out)),
        "n_with_assigned_treatment": int(out["AssignedTreatment"].fillna("").ne("").sum()),
        "single_dcbc_fraction_threshold": threshold,
        "barcode_count_table": str(filtered_count_path),
        "total_barcode_umi_definition": "Corrected DCBC UMI after read-support low-edge filtering.",
        "dendritic_cell_metadata": str(dirs["data"] / "dendritic_cell_metadata.csv"),
    }
    write_json(summary, dirs["qc"] / "dendritic_cell_metadata_summary.json")


def save_dual(fig: plt.Figure, name: str, dirs: dict[str, Path]) -> tuple[Path, Path]:
    pdf = dirs["fig_pdf"] / f"{name}.pdf"
    png = dirs["fig_png"] / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def treatment_color(config: dict[str, Any], treatment: str) -> str:
    return config["palette"].get(treatment, "#b7b7b7")


def violin(ax: plt.Axes, data: list[np.ndarray], positions: list[float], colors: list[str], widths: float = 0.7) -> None:
    if not data:
        return
    parts = ax.violinplot(data, positions=positions, widths=widths, showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#555555")
        body.set_linewidth(0.4)
        body.set_alpha(0.95)
    for arr, pos in zip(data, positions):
        if len(arr):
            ax.plot([pos - widths * 0.25, pos + widths * 0.25], [np.median(arr), np.median(arr)], color="#333333", linewidth=0.7)


def generate_figures(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    script_path = root / "code" / "scripts" / "generate_publication_figures.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing publication figure script: {script_path}")
    logger.info("Running replacement publication figure generator: %s", script_path)
    runpy.run_path(str(script_path), run_name="__main__")


def generate_umap_figure(
    config: dict[str, Any],
    root: Path,
    dirs: dict[str, Path],
    manifest: list[dict[str, str]],
    cell_meta: pd.DataFrame,
    t_meta: pd.DataFrame,
) -> None:
    analysis_path = resolve(root, config["paths"]["cellranger_analysis"])
    umap = read_analysis_csv(analysis_path, "umap/gene_expression_2_components/projection.csv")
    retained = cell_meta[cell_meta["cell_class"].isin(["dendritic_cell", "t_cell"])].copy()
    t_type = dict(zip(t_meta["CellBC"], t_meta["t_cell_type"]))
    labels = []
    for row in retained.itertuples(index=False):
        if row.cell_class == "dendritic_cell":
            labels.append("dendritic cell")
        else:
            labels.append("OTI T cell" if t_type.get(row.CellBC) == "OTI" else "C57BL/6 T cell")
    retained["plot_label"] = labels
    plot_df = retained.merge(umap, left_on="CellBC", right_on="Barcode", how="inner")
    out_table = dirs["tables"] / "figure_umap_retained_cells_source.csv"
    plot_df.to_csv(out_table, index=False)

    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    colors = {"dendritic cell": "#8fb3d9", "OTI T cell": "#f6b26b", "C57BL/6 T cell": "#93c47d"}
    order = ["dendritic cell", "C57BL/6 T cell", "OTI T cell"]
    for label in order:
        group = plot_df[plot_df["plot_label"] == label]
        ax.scatter(group["UMAP-1"], group["UMAP-2"], s=2, alpha=0.65, linewidths=0, color=colors[label], label=label, rasterized=True)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=5, loc="best")
    pdf, png = save_dual(fig, "gex_umap_retained_classified_cells", dirs)
    manifest.append(
        {
            "figure_panel": "Unsupervised GEX UMAP",
            "description": "CellRanger GEX UMAP subset to singlet retained classified cells; labels from marker classification and VDJ OTI call.",
            "source_table": str(out_table),
            "script": "src/publication_analysis/workflow.py::generate_umap_figure",
            "pdf": str(pdf),
            "png": str(png),
        }
    )


def generate_dc_violin(config: dict[str, Any], dirs: dict[str, Path], manifest: list[dict[str, str]], dc_meta: pd.DataFrame) -> None:
    df = dc_meta[dc_meta["AssignedTreatment"].fillna("").ne("")].copy()
    out_table = dirs["tables"] / "figure_dc_barcode_umi_by_treatment_source.csv"
    df.to_csv(out_table, index=False)
    treatments = [t for t in config["orders"]["treatments"] if t in set(df["AssignedTreatment"])]
    data = [np.log10(df.loc[df["AssignedTreatment"] == t, "total_barcode_umi"].to_numpy(dtype=float) + 1) for t in treatments]
    fig, ax = plt.subplots(figsize=(2.4, 1.9))
    violin(ax, data, list(range(len(treatments))), [treatment_color(config, t) for t in treatments])
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(treatments, rotation=35, ha="right")
    ax.set_ylabel("log10(total barcode UMI + 1)")
    ax.set_xlabel("Assigned treatment")
    pdf, png = save_dual(fig, "dc_total_barcode_umi_by_treatment_violin", dirs)
    manifest.append(
        {
            "figure_panel": "DC barcode UMI violin",
            "description": "Total barcode UMI per dendritic cell grouped by assigned treatment.",
            "source_table": str(out_table),
            "script": "src/publication_analysis/workflow.py::generate_dc_violin",
            "pdf": str(pdf),
            "png": str(png),
        }
    )


def generate_t_violin(config: dict[str, Any], dirs: dict[str, Path], manifest: list[dict[str, str]], t_meta: pd.DataFrame) -> None:
    out_table = dirs["tables"] / "figure_t_cell_barcode_umi_by_type_source.csv"
    t_meta.to_csv(out_table, index=False)
    groups = ["OTI", "C57BL6"]
    data = [np.log10(t_meta.loc[t_meta["t_cell_type"] == g, "total_barcode_umi"].to_numpy(dtype=float) + 1) for g in groups]
    fig, ax = plt.subplots(figsize=(1.8, 1.9))
    violin(ax, data, list(range(len(groups))), [config["palette"]["OTI"], config["palette"]["C57BL6"]], widths=0.6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["OTI", "C57BL/6"])
    ax.set_ylabel("log10(total barcode UMI + 1)")
    pdf, png = save_dual(fig, "t_cell_total_barcode_umi_by_type_violin", dirs)
    manifest.append(
        {
            "figure_panel": "T cell barcode UMI violin",
            "description": "Total barcode UMI per T cell grouped by OTI versus C57BL/6.",
            "source_table": str(out_table),
            "script": "src/publication_analysis/workflow.py::generate_t_violin",
            "pdf": str(pdf),
            "png": str(png),
        }
    )


def generate_peptide_violin(config: dict[str, Any], dirs: dict[str, Path], manifest: list[dict[str, str]], t_meta: pd.DataFrame) -> None:
    raw_path = downstream_barcode_count_path(dirs)
    t_cells = set(t_meta["CellBC"].astype(str))
    type_map = dict(zip(t_meta["CellBC"].astype(str), t_meta["t_cell_type"].astype(str)))
    counts: Counter[tuple[str, str]] = Counter()
    for chunk in pd.read_csv(raw_path, chunksize=500000, dtype={"CellBC": str, "PeptideBC_Name": str}):
        chunk = chunk[chunk["CellBC"].isin(t_cells)]
        if chunk.empty:
            continue
        grouped = chunk.groupby(["CellBC", "PeptideBC_Name"], as_index=False)["UMI"].sum()
        for row in grouped.itertuples(index=False):
            counts[(row.CellBC, row.PeptideBC_Name)] += int(row.UMI)
    rows = []
    peptide_order = config["orders"]["peptides"]
    for cell in sorted(t_cells):
        for peptide in peptide_order:
            umi = counts.get((cell, peptide), 0)
            rows.append({"CellBC": cell, "t_cell_type": type_map.get(cell, ""), "PeptideBC_Name": peptide, "UMI": umi})
    df = pd.DataFrame(rows)
    out_table = dirs["tables"] / "figure_t_cell_peptide_specific_umi_source.csv"
    df.to_csv(out_table, index=False)

    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    positions = []
    data = []
    colors = []
    xticks = []
    for i, peptide in enumerate(peptide_order):
        xticks.append(i)
        for offset, group, color in [(-0.18, "OTI", config["palette"]["OTI"]), (0.18, "C57BL6", config["palette"]["C57BL6"])]:
            values = df.loc[(df["PeptideBC_Name"] == peptide) & (df["t_cell_type"] == group), "UMI"].to_numpy(dtype=float)
            data.append(np.log10(values + 1))
            positions.append(i + offset)
            colors.append(color)
    violin(ax, data, positions, colors, widths=0.28)
    ax.set_xticks(xticks)
    ax.set_xticklabels(peptide_order, rotation=45, ha="right")
    ax.set_ylabel("log10(peptide UMI + 1)")
    ax.set_xlabel("Peptide")
    ax.plot([], [], color=config["palette"]["OTI"], label="OTI")
    ax.plot([], [], color=config["palette"]["C57BL6"], label="C57BL/6")
    ax.legend(fontsize=5, loc="upper right")
    pdf, png = save_dual(fig, "t_cell_peptide_specific_umi_violin", dirs)
    manifest.append(
        {
            "figure_panel": "Peptide-specific barcode UMI violin",
            "description": "Per-cell peptide-specific UMI for OTI and C57BL/6 T cells in requested peptide order.",
            "source_table": str(out_table),
            "script": "src/publication_analysis/workflow.py::generate_peptide_violin",
            "pdf": str(pdf),
            "png": str(png),
        }
    )


def generate_tcell_dcbc_bubble(
    config: dict[str, Any],
    dirs: dict[str, Path],
    manifest: list[dict[str, str]],
    cell_meta: pd.DataFrame,
    t_meta: pd.DataFrame,
) -> None:
    count_path = downstream_barcode_count_path(dirs)
    identity_map = load_dcbc_identity_map(dirs["data"] / "dcbc_identity_table.csv")
    class_map = dict(zip(cell_meta["CellBC"].astype(str), cell_meta["cell_class"].astype(str)))
    t_type_map = dict(zip(t_meta["CellBC"].astype(str), t_meta["t_cell_type"].astype(str)))

    donor_dcbc_by_condition: dict[tuple[str, str], set[str]] = defaultdict(set)
    t_condition_umi: Counter[tuple[str, str, str, str]] = Counter()
    for chunk in pd.read_csv(count_path, chunksize=500000, dtype={"CellBC": str, "DCBC": str}):
        grouped = chunk.groupby(["CellBC", "DCBC"], as_index=False)["UMI"].sum()
        for row in grouped.itertuples(index=False):
            identity = identity_map.get(row.DCBC)
            if not identity:
                continue
            peptide = identity.get("AssignedPeptideBC_Name", "")
            treatment = identity.get("AssignedTreatment", "")
            if not peptide or not treatment:
                continue
            cell_class = class_map.get(row.CellBC, "")
            if cell_class == "dendritic_cell":
                donor_dcbc_by_condition[(peptide, treatment)].add(row.DCBC)
            elif cell_class == "t_cell":
                t_type = t_type_map.get(row.CellBC, "")
                if t_type:
                    t_condition_umi[(row.CellBC, t_type, peptide, treatment)] += int(row.UMI)

    qualifying: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for (_cell, t_type, peptide, treatment), umi in t_condition_umi.items():
        if int(umi) >= 2:
            qualifying[(t_type, peptide, treatment)].append(int(umi))

    treatment_order = config["orders"]["treatments"]
    peptide_order = config["orders"]["peptides"]
    denominator_rows = []
    for peptide in peptide_order:
        for treatment in treatment_order:
            denominator_rows.append(
                {
                    "PeptideBC_Name": peptide,
                    "AssignedTreatment": treatment,
                    "donor_unique_dcbc": int(len(donor_dcbc_by_condition.get((peptide, treatment), set()))),
                }
            )
    denominator = pd.DataFrame(denominator_rows)
    denominator_path = dirs["tables"] / "donor_dcbc_denominator_by_condition.csv"
    denominator.to_csv(denominator_path, index=False)

    rows = []
    for t_type in ["OTI", "C57BL6"]:
        for peptide_i, peptide in enumerate(peptide_order):
            for treatment_i, treatment in enumerate(treatment_order):
                umis = qualifying.get((t_type, peptide, treatment), [])
                donor_unique = int(len(donor_dcbc_by_condition.get((peptide, treatment), set())))
                qualifying_cells = int(len(umis))
                geomean = float(np.exp(np.mean(np.log(np.asarray(umis, dtype=float))))) if umis else 0.0
                normalized = float(qualifying_cells / donor_unique) if donor_unique else 0.0
                rows.append(
                    {
                        "t_cell_type": t_type,
                        "PeptideBC_Name": peptide,
                        "AssignedTreatment": treatment,
                        "peptide_order": int(peptide_i),
                        "treatment_order": int(treatment_i),
                        "qualifying_cells": qualifying_cells,
                        "donor_unique_dcbc": donor_unique,
                        "qualifying_cells_per_donor_unique_dcbc": normalized,
                        "geomean_condition_umi": geomean,
                        "cell_condition_min_umi": 2,
                    }
                )
    bubble = pd.DataFrame(rows)
    out_table = dirs["tables"] / "tcell_dcbc_geomean_donor_normalized_bubble_source.csv"
    bubble.to_csv(out_table, index=False)

    positive_geomean = bubble.loc[bubble["geomean_condition_umi"] > 0, "geomean_condition_umi"].to_numpy(dtype=float)
    if len(positive_geomean):
        color_min = float(np.percentile(positive_geomean, 5))
        color_max = float(np.percentile(positive_geomean, 90))
        if color_max <= color_min:
            color_max = color_min + 1e-9
    else:
        color_min = 0.0
        color_max = 1.0
    size_max = float(bubble["qualifying_cells_per_donor_unique_dcbc"].max()) if len(bubble) else 0.0
    cmap = LinearSegmentedColormap.from_list("nt475_transfer_red", ["#f7fbff", "#f9d6d6", "#ed8590", "#c83f58"])
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 4.8), sharey=True)
    for ax, t_type in zip(axes, ["OTI", "C57BL6"]):
        sub = bubble[bubble["t_cell_type"] == t_type].copy()
        plot_sub = sub[sub["geomean_condition_umi"] > 0].copy()
        if not plot_sub.empty:
            ratio = plot_sub["qualifying_cells_per_donor_unique_dcbc"].to_numpy(dtype=float)
            sizes = 18 + (ratio / max(size_max, 1e-12)) * 230
            color_values = np.clip(plot_sub["geomean_condition_umi"].to_numpy(dtype=float), color_min, color_max)
            ax.scatter(
                plot_sub["treatment_order"],
                plot_sub["peptide_order"],
                s=sizes,
                c=color_values,
                cmap=cmap,
                vmin=color_min,
                vmax=color_max,
                edgecolors="#7a2f36",
                linewidths=0.35,
                alpha=0.92,
            )
        ax.set_title(t_type, fontsize=8)
        ax.set_xticks(range(len(treatment_order)))
        ax.set_xticklabels(treatment_order, rotation=35, ha="right")
        ax.set_xlim(-0.55, len(treatment_order) - 0.45)
        ax.set_ylim(len(peptide_order) - 0.45, -0.55)
        ax.grid(True, color="#e8e8e8", linewidth=0.35)
    axes[0].set_yticks(range(len(peptide_order)))
    axes[0].set_yticklabels(peptide_order)
    axes[0].set_ylabel("Peptide")
    for ax in axes:
        ax.set_xlabel("Treatment")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=color_min, vmax=color_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.03, extend="both")
    cbar.set_label("Geomean UMI", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    if size_max > 0:
        legend_values = [v for v in [0.25, 0.5, 1.0] if v <= size_max]
        if not legend_values:
            legend_values = [size_max]
        handles = [
            ax.scatter([], [], s=18 + (value / max(size_max, 1e-12)) * 230, color="#f4a6a6", edgecolors="#7a2f36", linewidths=0.35)
            for value in legend_values
        ]
        axes[-1].legend(handles, [f"{value:g}" for value in legend_values], title="Cells/DCBC", fontsize=5, title_fontsize=5, loc="lower right")
    pdf, png = save_dual(fig, "tcell_dcbc_geomean_donor_normalized_bubble", dirs)

    summary = {
        "source_count_table": str(count_path),
        "source_table": str(out_table),
        "denominator_table": str(denominator_path),
        "cell_condition_min_umi": 2,
        "treatment_order": treatment_order,
        "peptide_order": peptide_order,
        "n_bubble_rows": int(len(bubble)),
        "n_nonzero_bubbles": int((bubble["geomean_condition_umi"] > 0).sum()),
        "max_geomean_condition_umi": float(bubble["geomean_condition_umi"].max()) if len(bubble) else 0.0,
        "color_metric": "geometric mean condition DCBC UMI per qualifying T cell",
        "color_scale": "linear",
        "color_limits_scope": "global across OTI and C57BL6 panels",
        "color_limits_source": "5th and 90th percentiles of positive plotted geomean_condition_umi entries",
        "color_min": color_min,
        "color_max": color_max,
        "color_values_clipped_for_visualization_only": True,
        "max_qualifying_cells_per_donor_unique_dcbc": size_max,
        "pdf": str(pdf),
        "png": str(png),
    }
    write_json(summary, dirs["qc"] / "tcell_dcbc_geomean_donor_normalized_bubble_summary.json")
    manifest.append(
        {
            "figure_panel": "Donor-normalized T cell DCBC transfer bubble matrix",
            "description": "T cell DCBC transfer by peptide and treatment; color is geometric mean condition UMI among qualifying cells and circle size is qualifying cells per observed donor DCBC.",
            "source_table": str(out_table),
            "script": "src/publication_analysis/workflow.py::generate_tcell_dcbc_bubble",
            "pdf": str(pdf),
            "png": str(png),
        }
    )


def generate_clonotype_figure(config: dict[str, Any], dirs: dict[str, Path], manifest: list[dict[str, str]], t_meta: pd.DataFrame) -> None:
    counts = (
        t_meta[t_meta["clonotype_id"].fillna("").ne("")]
        .groupby("clonotype_id", as_index=False)
        .size()
        .rename(columns={"size": "n_t_cells"})
        .sort_values("n_t_cells", ascending=False)
        .head(20)
    )
    out_table = dirs["tables"] / "figure_top20_tcr_clonotypes_source.csv"
    counts.to_csv(out_table, index=False)
    fig_height = max(2.0, 0.13 * max(1, len(counts)) + 0.5)
    fig, ax = plt.subplots(figsize=(3.0, fig_height))
    y = np.arange(len(counts))
    ax.barh(y, counts["n_t_cells"], color="#a8d5ba", linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(counts["clonotype_id"], fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("T cells")
    ax.set_ylabel("TCR clonotype")
    pdf, png = save_dual(fig, "top20_tcr_clonotype_counts", dirs)
    manifest.append(
        {
            "figure_panel": "Top 20 TCR clonotype counts",
            "description": "Top 20 VDJ raw clonotype IDs among retained classified T cells.",
            "source_table": str(out_table),
            "script": "src/publication_analysis/workflow.py::generate_clonotype_figure",
            "pdf": str(pdf),
            "png": str(png),
        }
    )


def average_expression_by_group(
    matrix: sparse.csc_matrix,
    gene_to_indices: dict[str, list[int]],
    barcodes: list[str],
    panel: list[str],
    groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[str]]:
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    library_size = np.asarray(matrix.sum(axis=0)).ravel()
    rows = []
    missing = []
    for gene in panel:
        if gene not in gene_to_indices:
            missing.append(gene)
            continue
        gene_idx = gene_to_indices[gene][0]
        row: dict[str, Any] = {"gene": gene}
        for group_name, group_cells in groups.items():
            idx = np.array([barcode_to_idx[bc] for bc in group_cells if bc in barcode_to_idx], dtype=int)
            if len(idx) == 0:
                row[group_name] = np.nan
                continue
            counts = matrix[gene_idx, idx].toarray().ravel().astype(np.float32)
            norm = counts / np.maximum(library_size[idx].astype(np.float32), 1.0) * 10000.0
            row[group_name] = float(np.log1p(norm).mean())
        rows.append(row)
    return pd.DataFrame(rows), missing


def zscore_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    z = df.copy()
    values = z[group_cols].to_numpy(dtype=float)
    means = np.nanmean(values, axis=1, keepdims=True)
    stds = np.nanstd(values, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    values = (values - means) / stds
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    z[group_cols] = values
    return z


def order_genes_by_clustering(z_df: pd.DataFrame, group_cols: list[str]) -> list[int]:
    if len(z_df) <= 2:
        return list(range(len(z_df)))
    x = z_df[group_cols].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    try:
        linkage = hierarchy.linkage(x, method="average", metric="euclidean")
        return hierarchy.leaves_list(linkage).tolist()
    except Exception:
        return list(range(len(z_df)))


def plot_heatmap(
    z_df: pd.DataFrame,
    group_cols: list[str],
    config: dict[str, Any],
    title: str,
    name: str,
    dirs: dict[str, Path],
) -> tuple[Path, Path]:
    ordered = z_df.reset_index(drop=True)
    data = ordered[group_cols].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "nt475_diverging",
        [config["heatmap_colors"]["low"], config["heatmap_colors"]["mid"], config["heatmap_colors"]["high"]],
    )
    height = max(3.2, min(18.0, 0.085 * max(1, len(ordered)) + 0.8))
    width = max(2.2, 0.42 * len(group_cols) + 1.2)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-2, vmax=2)
    ax.set_xticks(range(len(group_cols)))
    ax.set_xticklabels(group_cols, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["gene"], fontsize=3.5)
    ax.set_title(title, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Row z-score", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    pdf, png = save_dual(fig, name, dirs)
    return pdf, png


def generate_heatmap_figures(
    config: dict[str, Any],
    root: Path,
    dirs: dict[str, Path],
    manifest: list[dict[str, str]],
    dc_meta: pd.DataFrame,
    t_meta: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    logger.info("Loading filtered GEX matrix for heatmaps")
    features, barcodes, matrix = load_10x_matrix(resolve(root, config["paths"]["filtered_feature_matrix"]))
    gene_to_indices = build_gene_index(features)

    treatment_order = config["orders"]["treatments"]
    dc_groups = {
        treatment: dc_meta.loc[dc_meta["AssignedTreatment"] == treatment, "CellBC"].astype(str).tolist()
        for treatment in treatment_order
    }
    dc_mean, dc_missing = average_expression_by_group(matrix, gene_to_indices, barcodes, config["dc_gene_panel"], dc_groups)
    dc_z = zscore_rows(dc_mean, treatment_order)
    dc_order = order_genes_by_clustering(dc_z, treatment_order)
    dc_mean = dc_mean.iloc[dc_order].reset_index(drop=True)
    dc_z = dc_z.iloc[dc_order].reset_index(drop=True)
    dc_mean_table = dirs["tables"] / "figure_dc_phenotype_heatmap_mean_expression_source.csv"
    dc_z_table = dirs["tables"] / "figure_dc_phenotype_heatmap_zscore_source.csv"
    dc_mean.to_csv(dc_mean_table, index=False)
    dc_z.to_csv(dc_z_table, index=False)
    pdf, png = plot_heatmap(dc_z, treatment_order, config, "DC phenotype", "dc_phenotype_treatment_heatmap", dirs)
    manifest.append(
        {
            "figure_panel": "DC phenotype heatmap",
            "description": "Mean log-normalized DC gene-panel expression by assigned treatment; genes clustered, columns fixed in biological order.",
            "source_table": str(dc_z_table),
            "script": "src/publication_analysis/workflow.py::generate_heatmap_figures",
            "pdf": str(pdf),
            "png": str(png),
        }
    )

    t_groups: dict[str, list[str]] = {}
    for treatment in treatment_order:
        label = f"{treatment} (OTI)"
        t_groups[label] = t_meta.loc[
            (t_meta["t_cell_type"] == "OTI")
            & (t_meta["interaction_state"] == "single_interaction")
            & (t_meta["AssignedTreatment"] == treatment),
            "CellBC",
        ].astype(str).tolist()
    t_groups["multi_interaction (OTI)"] = t_meta.loc[
        (t_meta["t_cell_type"] == "OTI") & (t_meta["interaction_state"] == "multi_interaction"),
        "CellBC",
    ].astype(str).tolist()
    t_groups["no_interaction (OTI)"] = t_meta.loc[
        (t_meta["t_cell_type"] == "OTI") & (t_meta["interaction_state"] == "no_interaction"),
        "CellBC",
    ].astype(str).tolist()
    t_groups["C57BL6 (all cells)"] = t_meta.loc[t_meta["t_cell_type"] == "C57BL6", "CellBC"].astype(str).tolist()
    t_group_order = config["orders"]["t_cell_heatmap_groups"]
    t_mean, t_missing = average_expression_by_group(matrix, gene_to_indices, barcodes, config["t_cell_gene_panel"], t_groups)
    t_z = zscore_rows(t_mean, t_group_order)
    t_order = order_genes_by_clustering(t_z, t_group_order)
    t_mean = t_mean.iloc[t_order].reset_index(drop=True)
    t_z = t_z.iloc[t_order].reset_index(drop=True)
    t_mean_table = dirs["tables"] / "figure_t_cell_phenotype_heatmap_mean_expression_source.csv"
    t_z_table = dirs["tables"] / "figure_t_cell_phenotype_heatmap_zscore_source.csv"
    t_mean.to_csv(t_mean_table, index=False)
    t_z.to_csv(t_z_table, index=False)
    pdf, png = plot_heatmap(t_z, t_group_order, config, "T cell phenotype", "t_cell_phenotype_group_heatmap", dirs)
    manifest.append(
        {
            "figure_panel": "T cell phenotype heatmap",
            "description": "Mean log-normalized T gene-panel expression by requested OTI/C57BL6 groups; genes clustered, columns fixed in biological order.",
            "source_table": str(t_z_table),
            "script": "src/publication_analysis/workflow.py::generate_heatmap_figures",
            "pdf": str(pdf),
            "png": str(png),
        }
    )
    write_json(
        {
            "dc_heatmap_missing_genes": dc_missing,
            "t_cell_heatmap_missing_genes": t_missing,
            "dc_group_cell_counts": {k: len(v) for k, v in dc_groups.items()},
            "t_cell_group_cell_counts": {k: len(v) for k, v in t_groups.items()},
        },
        dirs["qc"] / "heatmap_qc.json",
    )


def write_software_versions(path: Path) -> None:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": getattr(sparse, "__version__", None) or __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
    }
    write_json(versions, path)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_readme(config: dict[str, Any], root: Path, dirs: dict[str, Path], logger: logging.Logger) -> None:
    preflight = read_json_if_exists(dirs["qc"] / "preflight_summary.json")
    doublets = read_json_if_exists(dirs["qc"] / "doublet_filtering_summary.json")
    barcode_qc = read_json_if_exists(dirs["qc"] / "barcode_parse_qc.json")
    classify = read_json_if_exists(dirs["qc"] / "classification_qc.json")
    dcbc_correction = read_json_if_exists(dirs["qc"] / "dcbc_correction_summary.json")
    dcbc = read_json_if_exists(dirs["qc"] / "dcbc_identity_summary.json")
    read_support = read_json_if_exists(dirs["qc"] / "read_support_filter_summary.json")
    figure_summary = read_json_if_exists(dirs["qc"] / "figure_generation_summary.json")
    tcell = read_json_if_exists(dirs["qc"] / "t_cell_metadata_summary.json")
    dc = read_json_if_exists(dirs["qc"] / "dendritic_cell_metadata_summary.json")
    orientation_probe = preflight.get("relay_barcode_orientation_probe", {})
    orientation_counts = orientation_probe.get("probe_counts", {})
    selected_orientation = orientation_probe.get("selected_orientation", "not run")
    selected_orientation_matches = orientation_counts.get(f"both_{selected_orientation}", "not run")

    readme = f"""# NT475 Publication Analysis

This folder is a clean, auditable analysis package for the NT475 10x co-culture experiment. It is built from the raw inputs configured in `config.yaml`; archived exploratory material in `../Test/` is not used.

## One-command reproduction

```bash
cd {root}
make all
```

Equivalent direct command:

```bash
python3 run_workflow.py --config config.yaml
```

## Workflow

1. `preflight`: hashes configured raw inputs, inspects FASTQ read lengths, confirms the 10x CellBC/UMI layout in R1 and the Relay cassette in R2, and records the CellRanger `-1` barcode suffix convention.
2. `doublets`: performs Scrublet-style simulated doublet scoring in CellRanger GEX PCA space and removes cells at or above the configured doublet-score quantile.
3. `barcode_parse`: parses paired barcode FASTQs with exact Relay anchor and spacer matching, auto-detects direct versus reverse-complement treatment/peptide barcode orientation, keeps singlet CellRanger cell barcodes only, removes invalid treatment/peptide barcodes, removes globally singleton DCBCs, and writes exact raw parser outputs. The raw barcode table and raw per-UMI SQLite table are preserved for audit.
4. `dcbc_identity_precorrection`: builds a preliminary raw DCBC identity table used only to guide DCBC error correction.
5. `dcbc_correction`: loads raw per-UMI rows from `barcode_umi_counts.sqlite`, recursively collapses within-cell DCBC clusters at Hamming distance `<= {config['thresholds'].get('dcbc_correction_max_hamming_distance', 2)}` only when preliminary assigned peptide name and treatment match, unions raw 10x UMISeq sets, sums reads, and discards corrected per-cell DCBC clusters with reads `<= {config['thresholds'].get('dcbc_correction_discard_cell_dcbc_reads_le', 10)}`.
6. `dcbc_identity`: rebuilds the final public DCBC identity table from `barcode_corrected_count_table.csv.gz`.
7. `classification`: classifies singlet GEX cells as dendritic cell, T cell, or ambiguous/other by DC and T marker module scores.
8. `read_support_filter`: builds corrected CellBC+DCBC edge tables, fits separate robust Huber log10(Reads) ~ log10(UMI) relationships for assigned T-cell and dendritic-cell edges, removes low-read-support outlier edges, and writes the clean filtered count table used by all downstream DCBC-dependent analyses.
9. `t_cell_metadata` and `dendritic_cell_metadata`: assign interaction metadata, OTI/C57BL6 labels, and DCBC-derived peptide/treatment identities from the corrected plus read-support-filtered count table.
10. `figures`: runs `scripts/generate_publication_figures.py` from saved source/intermediate tables, including the DC-supported normalized transfer plots and DC treatment DGE additions, and exports every panel as both PDF and PNG.

## Key Thresholds

- Doublet threshold: score quantile `{config['thresholds']['doublet_score_quantile']}`.
- DCBC identity: dominant UMI fraction `>= {config['thresholds']['dcbc_identity_dominance_fraction']}`.
- DCBC correction: maximum Hamming distance `{config['thresholds'].get('dcbc_correction_max_hamming_distance', 2)}`; discard corrected per-cell DCBC clusters with reads `<= {config['thresholds'].get('dcbc_correction_discard_cell_dcbc_reads_le', 10)}`.
- Read-support edge filter: enabled `{config['thresholds'].get('read_support_filter_enabled', True)}`; Huber tuning `{config['thresholds'].get('read_support_huber_tuning', 1.345)}`; flag assigned T-cell/DC edges with UMI `>= {config['thresholds'].get('read_support_min_umi_for_outlier', 10)}` and residual below median minus `{config['thresholds'].get('read_support_mad_multiplier', 4.5)}` MAD sigma.
- DC-supported transfer plots: CellBC + peptide + treatment conditions require UMI `>= {config['thresholds'].get('bubble_min_condition_umi', 2)}` and are normalized by dendritic-cell supported DCBC UMI for the same peptide+treatment.
- T cell peptide/treatment assignment: summed assigned-DCBC UMI fraction `>= {config['thresholds'].get('t_cell_peptide_treatment_fraction', 0.85)}` for peptide and independently for treatment. Barcode-positive T cells without an assigned treatment are called `multi_interaction`; barcode-zero T cells remain `no_interaction`.
- Dendritic cell assignment: top DCBC fraction `> {config['thresholds']['dc_single_dcbc_fraction']}`.
- DC treatment DGE: dendritic cells require at least `{config['thresholds'].get('dc_dge_treatment_purity_threshold', 0.9) * 100:g}%` of assigned DCBC UMI from a single treatment; LPS and IFNg groups are compared with no-treatment controls using per-gene Welch tests on log1p(CP10k)-normalized expression. Genes require total contrast counts `>= {config['thresholds'].get('dc_dge_min_total_counts', 10)}` and detection in at least `{config['thresholds'].get('dc_dge_min_pct', 0.05)}` of either treatment or control cells; effect sizes are log2 fold-changes of mean CP10k expression with a pseudocount.
- Cell classification: marker score minimum `{config['thresholds']['marker_min_score']}` and score margin `{config['thresholds']['marker_score_margin']}`.

## Relay Barcode Orientation

Preflight selected `{selected_orientation}` orientation for treatment and peptide barcode matching. In the orientation probe, `{selected_orientation_matches}` read pairs matched both treatment and peptide barcode lists in the selected orientation. The observed R2 barcode segments are mapped back to canonical treatment barcode sequences from `config.yaml` and peptide identities from `OTI-Peptide-BC.csv`; therefore `data_intermediate/barcode_raw_count_table.csv.gz` reports canonical `TreatmentBC` values.

## Current Run Summary

- CellRanger filtered cells: `{preflight.get('cellranger_filtered_cells', 'not run')}`.
- Singlets retained after doublet filtering: `{doublets.get('n_singlets', 'not run')}`.
- Doublets removed: `{doublets.get('n_doublets', 'not run')}`.
- Final barcode read pairs after all filters: `{barcode_qc.get('final_read_pairs_after_all_filters_total', 'not run')}`.
- Collapsed barcode feature rows: `{barcode_qc.get('unique_feature_rows', 'not run')}`.
- Corrected cell-DCBC pairs after Hamming collapse: `{dcbc_correction.get('after_hamming_cell_dcbc_pairs', 'not run')}`.
- Corrected cell-DCBC pairs after reads filter: `{dcbc_correction.get('after_read_filter_cell_dcbc_pairs', 'not run')}`.
- DCBC correction collapse events: `{dcbc_correction.get('collapse_events', 'not run')}`.
- Read-support filtered count rows: `{read_support.get('filtered_count_rows', 'not run')}`.
- Read-support flagged edges total: `{read_support.get('flagged_edges_total', 'not run')}`.
- Final publication figures: `{figure_summary.get('n_figures', 'not run')}`.
- Assigned DCBC identities: `{dcbc.get('n_assigned', 'not run')}`; ambiguous DCBCs: `{dcbc.get('n_ambiguous', 'not run')}`.
- Cell classifications: `{classify.get('classification_counts', 'not run')}`.
- T cell type counts: `{tcell.get('t_cell_type_counts', 'not run')}`.
- T cell interaction states: `{tcell.get('interaction_state_counts', 'not run')}`.
- Dendritic cells with assigned treatment: `{dc.get('n_with_assigned_treatment', 'not run')}`.

## Important Outputs

- Raw input hashes: `data_intermediate/raw_input_manifest.csv`.
- Preflight QC: `results/qc/preflight_summary.json`.
- Doublet outputs: `results/tables/singlet_barcodes.csv`, `results/tables/doublet_barcodes.csv`, `results/tables/doublet_scores.csv`, `results/qc/doublet_filtering_summary.json`.
- Collapsed barcode raw count table: `data_intermediate/barcode_raw_count_table.csv.gz`.
- Raw per-UMI parser table: `data_intermediate/barcode_umi_counts.sqlite`.
- Preliminary raw DCBC identity: `data_intermediate/dcbc_identity_precorrection_table.csv`, `data_intermediate/dcbc_identity_precorrection_components.csv`, `results/qc/dcbc_identity_precorrection_summary.json`.
- Corrected barcode count table preserved before read-support filtering: `data_intermediate/barcode_corrected_count_table.csv.gz`.
- DCBC correction audit outputs: `data_intermediate/dcbc_correction_events.csv.gz`, `results/qc/dcbc_correction_summary.json`.
- Read-support filtered barcode count table used by downstream DCBC analyses: `data_intermediate/barcode_read_support_filtered_count_table.csv.gz`.
- Read-support audit outputs: `data_intermediate/read_support_edge_table.csv.gz`, `data_intermediate/read_support_low_support_edges.csv`, `results/qc/read_support_filter_summary.json`, `figures/pdf/read_support_tcell_vs_dendritic_cell.pdf`, `figures/png/read_support_tcell_vs_dendritic_cell.png`.
- Barcode parse QC: `results/qc/barcode_parse_qc.json`.
- Final corrected DCBC identity table: `data_intermediate/dcbc_identity_table.csv`.
- Cell metadata: `data_intermediate/cell_metadata.csv`.
- T cell metadata: `data_intermediate/t_cell_metadata.csv`.
- Dendritic cell metadata: `data_intermediate/dendritic_cell_metadata.csv`.
- Figure source tables and final figure manifest: `results/tables/figure_manifest.csv`.
- Figure generation QC summary: `results/qc/run_summary.json`, `results/qc/figure_generation_summary.json`.
- DC-supported normalized T cell DCBC transfer bubble matrices: `results/tables/bubble_dc_supported_normalized_source.csv`, `results/tables/bubble_dc_supported_denominator_by_condition.csv`, `results/tables/bubble_dc_supported_normalized_qualifying_cell_conditions.csv`, `results/qc/bubble_dc_supported_normalized_summary.json`.
- DC-supported normalized T cell treatment boxplots: `results/tables/tcell_treatment_dc_supported_normalized_peptide_condition_source.csv`, `results/tables/tcell_treatment_dc_supported_normalized_peptide_collapsed_single_cell_source.csv`, `results/tables/tcell_treatment_dc_supported_normalized_stats_vs_no_treatment.csv`, `results/qc/tcell_treatment_dc_supported_normalized_box_summary.json`.
- DC treatment DGE: `results/tables/dc_treatment_assignment_90pct_purity.csv`, `results/tables/dc_dge_cells_used.csv`, `results/tables/dc_dge_welch_results.csv.gz`, `results/tables/dc_dge_program_summary.csv`, `results/tables/dc_dge_top_gene_heatmap_source.csv`, `results/tables/dc_dge_volcano_LPS_vs_no_treatment_source.csv`, `results/tables/dc_dge_volcano_IFNg_vs_no_treatment_source.csv`, `results/qc/dc_treatment_dge_summary.json`.
- DC primary DGE heatmap: `results/tables/dc_dge_primary_gene_selection.csv`, `results/tables/dc_dge_top_gene_heatmap_primary_source.csv`, `results/tables/dc_dge_primary_gene_contrast_metrics.csv`, `results/qc/dc_dge_top_gene_heatmap_primary_summary.json`.
- T cell signature updates: `results/tables/t_cell_signature_group_median_heatmap_zscore.csv`, `results/tables/tcell_signature_effect_ci_vs_no_interaction_source.csv`, `results/tables/tcell_highlighted_gene_relative_pattern_bubble_source.csv`, `results/qc/t_cell_signature_figure_updates_summary.json`.
- Publication figures: `figures/pdf/` and `figures/png/`.
- QC plots: `results/qc/plots/`.
- Software versions: `results/qc/software_versions.json`.
- Full run log: `logs/workflow.log`.

## Figure Reproducibility

Every figure is generated from a saved table in `results/tables/`. The mapping from figure panel to source table, script function, PDF, and PNG is written to `results/tables/figure_manifest.csv`.

## Plot Formatting

Figures use Matplotlib PDF TrueType font embedding (`pdf.fonttype=42`, `ps.fonttype=42`), clean axes, compact panel dimensions, and the pastel palette specified in `config.yaml`. Heatmap columns are kept in the configured biological order; only genes/rows are clustered.
"""
    readme_path = dirs["qc"] / "full_workflow_outputs.md"
    readme_path.write_text(readme, encoding="utf-8")
    logger.info("Wrote %s", readme_path)
