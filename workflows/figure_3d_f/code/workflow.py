#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shlex
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FASTQ_DIR = RAW_DIR / "fastq"
REFERENCE_DIR = RAW_DIR / "reference"
FLOW_EXPORT_DIR = RAW_DIR / "flow_exports"
MAGECK_DIR = ROOT / "data" / "upstream" / "mageck"
FIGURE_TABLE_DIR = ROOT / "data" / "figure_tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
SUMMARY_DIR = ROOT / "outputs" / "summaries"

for directory in [FIGURE_TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

matplotlib_cache = Path(tempfile.gettempdir()) / "figure_3de_matplotlib_cache"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.use14corefonts"] = False
mpl.rcParams["text.usetex"] = False
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"

PEPTIDES_FILE = REFERENCE_DIR / "Peptides.txt"
IFN_FILE = REFERENCE_DIR / "IFN.csv"
LFC_OUT = FIGURE_TABLE_DIR / "LFC.csv"

PEPTIDES = [
    "N4",
    "A2",
    "Y3",
    "Q4",
    "T4",
    "V4",
    "G4",
    "D4",
    "Q4R7",
    "Q4H7",
    "Q7",
    "E1",
    "CATNB",
    "LCMV",
    "TB",
    "MCMV",
]
CONTROLS = ("CATNB", "LCMV", "TB", "MCMV")
DEFAULT_COLOR = "#84c7ff"
CONTROL_COLOR = "#7a716b"
CUSTOM_PEPTIDE_COLORS = {peptide: DEFAULT_COLOR for peptide in PEPTIDES}
for control in CONTROLS:
    CUSTOM_PEPTIDE_COLORS[control] = CONTROL_COLOR


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
        return {str(key): clean_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(clean_for_json(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_raw_manifest() -> list[dict[str, Any]]:
    entries = []
    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".gz":
            role = "sequencing_fastq"
        elif path.suffix.lower() == ".fcs":
            role = "flow_cytometry_fcs"
        elif path.name in {"Peptides.txt", "IFN.csv"}:
            role = "figure_input_table"
        else:
            role = "raw_or_exported_input"
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


def output_file_records() -> list[dict[str, Any]]:
    records = []
    for root in [ROOT / "data" / "intermediate", FIGURE_TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]:
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


def extract_mageck_commands() -> list[dict[str, str]]:
    rows = []
    for log_path in sorted(MAGECK_DIR.glob("*.log")):
        with log_path.open(errors="replace") as handle:
            for line in handle:
                match = re.search(r"Parameters:\s*(.+)$", line)
                if not match:
                    continue
                command = match.group(1).strip()
                command = re.sub(r"\S*/mageck\b", "mageck", command)
                rows.append(
                    {
                        "log_file": relpath(log_path),
                        "output_prefix": log_path.stem,
                        "command": command,
                    }
                )
                break

    command_df = pd.DataFrame(rows)
    command_df.to_csv(SUMMARY_DIR / "mageck_commands.csv", index=False)
    write_json(SUMMARY_DIR / "mageck_commands.json", {"commands": rows})
    write_mageck_rerun_script(rows)
    return rows


def fastq_label_from_actual(path: Path) -> str:
    name = path.name
    if name.endswith(".fastq.gz"):
        stem = name[:-9]
    elif name.endswith(".fq.gz"):
        stem = name[:-6]
    else:
        stem = path.stem
    return stem.split("_")[0]


def write_fastq_label_manifest() -> pd.DataFrame:
    countsummary = pd.read_csv(MAGECK_DIR / "OTI.countsummary.txt", sep="\t")
    raw_fastqs = sorted(FASTQ_DIR.glob("*.fastq.gz"), key=lambda path: path.name)
    actual_by_label = {fastq_label_from_actual(path): path.name for path in raw_fastqs}

    rows = []
    for _, row in countsummary.iterrows():
        logged_file = str(row["File"])
        label = str(row["Label"])
        logged_label = logged_file.replace(".fastq.gz", "")
        rows.append(
            {
                "mageck_label": label,
                "logged_fastq": logged_file,
                "raw_fastq": actual_by_label.get(logged_label, ""),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(SUMMARY_DIR / "fastq_label_manifest.csv", index=False)
    return manifest


def write_mageck_rerun_script(commands: list[dict[str, str]]) -> None:
    manifest_path = SUMMARY_DIR / "fastq_label_manifest.csv"
    script_path = ROOT / "run_mageck_commands.sh"
    count_commands = [row["command"] for row in commands if " mageck count " in f" {row['command']} "]
    test_commands = [row["command"] for row in commands if " mageck test " in f" {row['command']} "]

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'WORKDIR="$ROOT/data/intermediate/mageck_rerun"',
        'mkdir -p "$WORKDIR"',
        'cd "$WORKDIR"',
        'cp "$ROOT/data/intermediate/mageck_output/OTI.txt" ./',
        "",
        "# Link raw FASTQs to the filenames recorded in the MAGeCK log.",
        f"# Mapping source: {relpath(manifest_path)}",
    ]

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        for _, row in manifest.iterrows():
            raw_fastq = row["raw_fastq"]
            logged_fastq = row["logged_fastq"]
            if isinstance(raw_fastq, str) and raw_fastq:
                lines.append(
                    f'ln -sf "$ROOT/data/raw/fastq/{shlex.quote(raw_fastq)}" {shlex.quote(logged_fastq)}'
                )

    lines.extend(["", "# Recovered MAGeCK commands."])
    for command in count_commands + test_commands:
        lines.append(command)

    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)


def derive_lfc_from_mageck() -> pd.DataFrame:
    peptides = pd.read_table(PEPTIDES_FILE, dtype=str)
    if "Peptide" not in peptides.columns:
        raise ValueError("Peptides.txt must contain a 'Peptide' column.")
    peptides["Peptide"] = peptides["Peptide"].astype(str).str.strip()
    result = peptides[["Peptide"]].copy()

    id_candidates = ["Peptide", "peptide", "id", "ID", "Gene", "gene"]
    summary_files = sorted(MAGECK_DIR.glob("*.gene_summary.txt"))
    if not summary_files:
        raise FileNotFoundError(f"No MAGeCK gene summary files found in {MAGECK_DIR}")

    for summary_path in summary_files:
        df = pd.read_csv(summary_path, sep="\t", dtype=str)
        if "pos|lfc" not in df.columns:
            raise ValueError(f"Missing 'pos|lfc' in {summary_path}")
        id_col = next((col for col in id_candidates if col in df.columns), None)
        if id_col is None:
            raise ValueError(f"No peptide/gene ID column found in {summary_path}")
        df[id_col] = df[id_col].astype(str).str.strip()
        lfc_map = (
            df[[id_col, "pos|lfc"]]
            .dropna(subset=[id_col])
            .drop_duplicates(subset=[id_col], keep="first")
            .set_index(id_col)["pos|lfc"]
        )
        result[summary_path.name.split(".")[0]] = result["Peptide"].map(lfc_map)

    result.to_csv(LFC_OUT, index=False)
    return result


def connected_components(norm_xy: np.ndarray, radius: float) -> list[list[int]]:
    seen = np.zeros(len(norm_xy), dtype=bool)
    clusters = []
    radius2 = radius * radius

    for i in range(len(norm_xy)):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        component = []
        while stack:
            j = stack.pop()
            component.append(j)
            dist2 = np.sum((norm_xy - norm_xy[j]) ** 2, axis=1)
            neighbors = np.where((dist2 <= radius2) & (~seen))[0]
            for neighbor in neighbors:
                seen[neighbor] = True
                stack.append(int(neighbor))
        clusters.append(component)
    return clusters


def seed_label_positions(
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cluster_radius_frac: float = 0.05,
    base_offset_frac: float = 0.035,
) -> np.ndarray:
    xspan = max(xlim[1] - xlim[0], 1e-12)
    yspan = max(ylim[1] - ylim[0], 1e-12)
    norm_xy = np.column_stack([(x - xlim[0]) / xspan, (y - ylim[0]) / yspan])
    seeded = np.column_stack([x.copy(), y.copy()])

    for component in connected_components(norm_xy, cluster_radius_frac):
        points = norm_xy[component]
        center_x, center_y = points.mean(axis=0)
        if len(component) == 1:
            vx = points[0, 0] - 0.5
            vy = points[0, 1] - 0.5
            if abs(vx) + abs(vy) < 1e-9:
                vx, vy = 1.0, 1.0
            norm = math.hypot(vx, vy)
            seeded[component[0], 0] = x[component[0]] + vx / norm * base_offset_frac * xspan
            seeded[component[0], 1] = y[component[0]] + vy / norm * base_offset_frac * yspan
            continue

        angles = np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x)
        order = np.array(component)[np.argsort(angles)]
        phase = math.atan2(center_y - 0.5, center_x - 0.5) + math.pi
        placed = 0
        ring = 0
        while placed < len(component):
            ring += 1
            capacity = 8 + 4 * (ring - 1)
            radius = base_offset_frac * (1.6 + 0.95 * (ring - 1))
            count = min(capacity, len(component) - placed)
            for slot in range(count):
                idx = order[placed + slot]
                angle = phase + 2 * math.pi * slot / count
                seeded[idx, 0] = xlim[0] + center_x * xspan + math.cos(angle) * radius * xspan
                seeded[idx, 1] = ylim[0] + center_y * yspan + math.sin(angle) * radius * yspan
            placed += count
    return seeded


def relax_labels(
    ax,
    texts,
    anchor_xy: np.ndarray,
    n_iter: int = 400,
    text_expand: tuple[float, float] = (1.03, 1.12),
    text_push: float = 1.0,
    point_push: float = 1.2,
    pull_strength: float = 0.015,
    point_padding_px: int = 10,
    max_step_px: int = 12,
    margin_px: int = 4,
) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    anchors_disp = ax.transData.transform(anchor_xy)
    inv = ax.transData.inverted()
    positions = ax.transData.transform(np.array([text.get_position() for text in texts]))

    for _ in range(n_iter):
        for text, position in zip(texts, positions):
            text.set_position(inv.transform(position))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [text.get_window_extent(renderer=renderer).expanded(*text_expand) for text in texts]
        delta = np.zeros_like(positions)

        for i, box_i in enumerate(bboxes):
            center_i = np.array([(box_i.x0 + box_i.x1) / 2, (box_i.y0 + box_i.y1) / 2])
            for j in range(i + 1, len(texts)):
                box_j = bboxes[j]
                if not box_i.overlaps(box_j):
                    continue
                center_j = np.array([(box_j.x0 + box_j.x1) / 2, (box_j.y0 + box_j.y1) / 2])
                overlap_x = min(box_i.x1, box_j.x1) - max(box_i.x0, box_j.x0)
                overlap_y = min(box_i.y1, box_j.y1) - max(box_i.y0, box_j.y0)
                direction = center_i - center_j
                if np.allclose(direction, 0):
                    direction = np.array([1.0 if (i + j) % 2 == 0 else -1.0, 1.0])
                if overlap_x < overlap_y:
                    direction[1] *= 0.35
                else:
                    direction[0] *= 0.35
                direction /= np.linalg.norm(direction)
                push = 0.5 * max(overlap_x, overlap_y) + 2.0
                delta[i] += direction * push * text_push
                delta[j] -= direction * push * text_push

        for i, bbox in enumerate(bboxes):
            point = anchors_disp[i]
            point_box = Bbox.from_extents(
                point[0] - point_padding_px,
                point[1] - point_padding_px,
                point[0] + point_padding_px,
                point[1] + point_padding_px,
            )
            if bbox.overlaps(point_box):
                center = np.array([(bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2])
                direction = center - point
                if np.allclose(direction, 0):
                    direction = np.array([1.0, 1.0])
                direction /= np.linalg.norm(direction)
                overlap_x = min(bbox.x1, point_box.x1) - max(bbox.x0, point_box.x0)
                overlap_y = min(bbox.y1, point_box.y1) - max(bbox.y0, point_box.y0)
                delta[i] += direction * (max(overlap_x, overlap_y, 0) + 2.0) * point_push

        vec = positions - anchors_disp
        dist = np.linalg.norm(vec, axis=1, keepdims=True)
        unit = np.divide(vec, np.maximum(dist, 1e-9))
        delta += np.where(dist > 18, -unit * (dist - 18) * pull_strength, 0.0)

        for i, bbox in enumerate(bboxes):
            if bbox.x0 < ax_bbox.x0 + margin_px:
                delta[i, 0] += ax_bbox.x0 + margin_px - bbox.x0
            if bbox.x1 > ax_bbox.x1 - margin_px:
                delta[i, 0] -= bbox.x1 - (ax_bbox.x1 - margin_px)
            if bbox.y0 < ax_bbox.y0 + margin_px:
                delta[i, 1] += ax_bbox.y0 + margin_px - bbox.y0
            if bbox.y1 > ax_bbox.y1 - margin_px:
                delta[i, 1] -= bbox.y1 - (ax_bbox.y1 - margin_px)

        norms = np.linalg.norm(delta, axis=1)
        delta *= np.minimum(1.0, max_step_px / np.maximum(norms, 1e-9))[:, None]
        positions += delta
        if len(norms) == 0 or norms.max() < 0.5:
            break

    for text, position in zip(texts, positions):
        text.set_position(inv.transform(position))
    fig.canvas.draw()


def peptide_color_map(peptides: list[str]) -> dict[str, str]:
    colors = {peptide: DEFAULT_COLOR for peptide in peptides}
    for control in CONTROLS:
        colors[control] = CONTROL_COLOR
    colors.update(CUSTOM_PEPTIDE_COLORS)
    return colors


def normalized_lfc_table(
    lfc_df: pd.DataFrame,
    sample_x: str,
    sample_y: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    normalize_stat: str = "mean",
) -> pd.DataFrame:
    reducer = np.nanmean if normalize_stat == "mean" else np.nanmedian
    work = lfc_df[["Peptide", sample_x, sample_y]].copy()
    work["Peptide"] = work["Peptide"].astype(str).str.strip()
    work = work[work["Peptide"].isin(PEPTIDES)].copy()
    work[sample_x] = pd.to_numeric(work[sample_x], errors="coerce")
    work[sample_y] = pd.to_numeric(work[sample_y], errors="coerce")
    work["Peptide"] = pd.Categorical(work["Peptide"], categories=PEPTIDES, ordered=True)
    work = work.sort_values("Peptide").reset_index(drop=True)
    work["Peptide"] = work["Peptide"].astype(str)

    control_mask = work["Peptide"].isin(CONTROLS)
    offset_x = reducer(work.loc[control_mask, sample_x].dropna().to_numpy(dtype=float))
    offset_y = reducer(work.loc[control_mask, sample_y].dropna().to_numpy(dtype=float))
    work["x"] = work[sample_x] - offset_x
    work["y"] = work[sample_y] - offset_y
    work = work.dropna(subset=["x", "y"]).copy()
    work["sample_x"] = sample_x
    work["sample_y"] = sample_y
    work["x_control_offset"] = offset_x
    work["y_control_offset"] = offset_y
    work["color"] = work["Peptide"].map(peptide_color_map(PEPTIDES)).fillna(DEFAULT_COLOR)
    work["xlim_min"] = xlim[0]
    work["xlim_max"] = xlim[1]
    work["ylim_min"] = ylim[0]
    work["ylim_max"] = ylim[1]
    return work


def plot_normalized_lfc(
    plot_df: pd.DataFrame,
    name: str,
    xlim: tuple[float, float] = (-1, 3),
    ylim: tuple[float, float] = (-1, 3),
    label_cluster_radius_frac: float = 0.06,
    label_relax_iterations: int = 500,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=200,
        c=plot_df["color"],
        alpha=0.85,
        edgecolors=plot_df["color"],
        linewidths=0,
        zorder=3,
    )
    ax.axvline(0, linestyle="--", linewidth=1, color="0.75", zorder=1)
    ax.axhline(0, linestyle="--", linewidth=1, color="0.75", zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_box_aspect(1)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 1))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 1))

    x = plot_df["x"].to_numpy(dtype=float)
    y = plot_df["y"].to_numpy(dtype=float)
    anchor_xy = np.column_stack([x, y])
    seeded_xy = seed_label_positions(
        x,
        y,
        xlim=xlim,
        ylim=ylim,
        cluster_radius_frac=label_cluster_radius_frac,
        base_offset_frac=0.035,
    )

    texts = [
        ax.text(sx, sy, row["Peptide"], fontsize=9, ha="center", va="center", zorder=5)
        for (_, row), (sx, sy) in zip(plot_df.iterrows(), seeded_xy)
    ]
    relax_labels(ax, texts, anchor_xy, n_iter=label_relax_iterations)

    label_rows = []
    for text, (_, row), (px, py) in zip(texts, plot_df.iterrows(), anchor_xy):
        tx, ty = text.get_position()
        ax.plot([px, tx], [py, ty], color="0.45", linewidth=0.8, zorder=2)
        label_rows.append({"Peptide": row["Peptide"], "label_x": tx, "label_y": ty})

    ax.set_xlabel(f"{plot_df['sample_x'].iloc[0]} (control-normalized LFC)")
    ax.set_ylabel(f"{plot_df['sample_y'].iloc[0]} (control-normalized LFC)")
    ax.set_title(f"{plot_df['sample_x'].iloc[0]} vs {plot_df['sample_y'].iloc[0]}")
    fig.tight_layout()

    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(SUMMARY_DIR / f"{name}_label_positions_generated.csv", index=False)

    png_path = FIGURE_DIR / f"{name}.png"
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": relpath(png_path), "pdf": relpath(pdf_path)}


def ifn_vs_ot_table(lfc_df: pd.DataFrame, ifn_df: pd.DataFrame, ylim: tuple[float, float]) -> pd.DataFrame:
    merged = pd.merge(lfc_df[["Peptide", "OT"]], ifn_df[["Peptide", "IFN"]], on="Peptide", how="inner")
    merged["Peptide"] = merged["Peptide"].astype(str).str.strip()
    merged = merged[merged["Peptide"].isin(PEPTIDES)].copy()
    merged["OT"] = pd.to_numeric(merged["OT"], errors="coerce")
    merged["IFN"] = pd.to_numeric(merged["IFN"], errors="coerce")
    merged = merged.dropna().copy()
    merged["Peptide"] = pd.Categorical(merged["Peptide"], categories=PEPTIDES, ordered=True)
    merged = merged.sort_values("Peptide").reset_index(drop=True)
    merged["Peptide"] = merged["Peptide"].astype(str)
    offset = np.nanmean(merged.loc[merged["Peptide"].isin(CONTROLS), "OT"].to_numpy(dtype=float))
    merged["OT_norm"] = merged["OT"] - offset
    merged["Color"] = merged["Peptide"].map(peptide_color_map(PEPTIDES))
    merged["OT_control_offset"] = offset
    merged["ylim_min"] = ylim[0]
    merged["ylim_max"] = ylim[1]
    return merged


def plot_ifn_vs_ot(
    plot_df: pd.DataFrame,
    ylim: tuple[float, float] = (-1, 3),
    label_cluster_radius_frac: float = 0.06,
    label_relax_iterations: int = 500,
) -> dict[str, str]:
    x = plot_df["IFN"].to_numpy(dtype=float)
    x_pad = max(0.25, np.nanmax(np.abs(x)) * 0.1)
    xlim = (np.nanmin(x) - x_pad, np.nanmax(x) + x_pad)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        plot_df["IFN"],
        plot_df["OT_norm"],
        s=200,
        c=plot_df["Color"],
        alpha=0.85,
        edgecolors="none",
        linewidths=0,
        zorder=3,
    )
    ax.axhline(0, linestyle="--", color="0.75")
    ax.axvline(0, linestyle="--", color="0.75")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_box_aspect(1)
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 1))

    y = plot_df["OT_norm"].to_numpy(dtype=float)
    anchor_xy = np.column_stack([x, y])
    seeded_xy = seed_label_positions(
        x,
        y,
        xlim=xlim,
        ylim=ylim,
        cluster_radius_frac=label_cluster_radius_frac,
        base_offset_frac=0.035,
    )

    texts = [
        ax.text(sx, sy, row["Peptide"], fontsize=9, ha="center", va="center", zorder=5)
        for (_, row), (sx, sy) in zip(plot_df.iterrows(), seeded_xy)
    ]
    relax_labels(ax, texts, anchor_xy, n_iter=label_relax_iterations)

    label_rows = []
    for text, (_, row), (px, py) in zip(texts, plot_df.iterrows(), anchor_xy):
        tx, ty = text.get_position()
        ax.plot([px, tx], [py, ty], color="0.5", lw=0.8, zorder=2)
        label_rows.append({"Peptide": row["Peptide"], "label_x": tx, "label_y": ty})

    ax.set_xlabel("IFN")
    ax.set_ylabel("OT (control-normalized LFC)")
    ax.set_title("IFN vs OT")
    fig.tight_layout()

    pd.DataFrame(label_rows).to_csv(SUMMARY_DIR / "IFN_vs_OT_label_positions_generated.csv", index=False)
    png_path = FIGURE_DIR / "IFN_vs_OT.png"
    pdf_path = FIGURE_DIR / "IFN_vs_OT.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": relpath(png_path), "pdf": relpath(pdf_path)}


def main() -> None:
    start_time = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    raw_manifest = write_raw_manifest()
    fastq_label_manifest = write_fastq_label_manifest()
    mageck_commands = extract_mageck_commands()
    lfc_df = derive_lfc_from_mageck()
    ifn_df = pd.read_csv(IFN_FILE, encoding="utf-8-sig")

    figure_outputs = {}
    figure_tables = {}
    for name, sample_x, sample_y in [
        ("OT", "OT-1", "OT-2"),
        ("WT", "WT-1", "WT-2"),
    ]:
        plot_df = normalized_lfc_table(lfc_df, sample_x, sample_y, xlim=(-1, 3), ylim=(-1, 3))
        table_path = FIGURE_TABLE_DIR / f"{name}_plot_table.csv"
        plot_df.to_csv(table_path, index=False)
        figure_tables[name] = relpath(table_path)
        figure_outputs[name] = plot_normalized_lfc(plot_df, name)

    ifn_plot_df = ifn_vs_ot_table(lfc_df, ifn_df, ylim=(-1, 3))
    ifn_table_path = FIGURE_TABLE_DIR / "IFN_vs_OT_plot_table.csv"
    ifn_plot_df.to_csv(ifn_table_path, index=False)
    figure_tables["IFN_vs_OT"] = relpath(ifn_table_path)
    figure_outputs["IFN_vs_OT"] = plot_ifn_vs_ot(ifn_plot_df, ylim=(-1, 3))

    qc_summary = {
        "mageck": {
            "commands_recovered_from_logs": len(mageck_commands),
            "command_table": "summaries/mageck_commands.csv",
            "rerun_script": "run_mageck_commands.sh",
            "fastq_label_manifest": "summaries/fastq_label_manifest.csv",
            "mageck_output_files": len([path for path in MAGECK_DIR.glob("*") if path.is_file()]),
        },
        "tables": {
            "LFC_rows": int(len(lfc_df)),
            "LFC_columns": int(len(lfc_df.columns)),
            "IFN_rows": int(len(ifn_df)),
            "fastq_label_rows": int(len(fastq_label_manifest)),
        },
        "figures": figure_outputs,
        "figure_tables": figure_tables,
    }
    write_json(SUMMARY_DIR / "qc_summary.json", qc_summary)

    run_summary = {
        "workflow": "Figure 3D-F MAGeCK LFC extraction and publication figure generation",
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
