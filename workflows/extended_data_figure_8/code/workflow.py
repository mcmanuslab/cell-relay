#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/extended_data_figure8_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import adjustText
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "outputs" / "summaries"
TABLE_DIR = ROOT / "data" / "figure_tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

FLOW_DIR = ROOT / "data" / "upstream" / "flow"
RNA_DIR = ROOT / "data" / "upstream" / "rnaseq"
BAM_DIR = RNA_DIR / "JTRWM7_bam"
ANNOTATION_DIR = RNA_DIR / "annotation"
COUNT_DIR = RNA_DIR / "gene_count_cache"

TABLE_PATH = ROOT / "table.tsv"
GTF_PATH = ANNOTATION_DIR / "Homo_sapiens.GRCh38.114.gtf.gz"
COUNT_MATRIX_PATH = RNA_DIR / "JTRWM7_gene_counts_by_index.tsv"

MIN_TOTAL_COUNTS = 20
LFC_THRESHOLD = 1.0
FDR_THRESHOLD = 0.05
TOP_LFC_LABELS_PER_SIDE = 5
PRIOR_CPM = 0.5
VOLCANO_LABEL_LAYOUT_SEED = 0
CUSTOM_LABEL_GENES_10_9 = ("CALR",)
BIN_SIZE = 16_384
COUNT_MULTIMAPPERS = False
SKIP_DUPLICATES = True

VOLCANO_UP_COLOR = "#ed8590"
VOLCANO_DOWN_COLOR = "#5d8af7"
VOLCANO_NEUTRAL_COLOR = "#aeb4bc"
VOLCANO_AXIS_COLOR = "#202124"
VOLCANO_GUIDE_COLOR = "#6f7378"
VOLCANO_GRID_COLOR = "#e8eaed"
VOLCANO_NONSIG_POINT_SIZE = 10
VOLCANO_SIG_POINT_SIZE = 22
VOLCANO_GENE_LABEL_SIZE = 9.5
VOLCANO_AXIS_LABEL_SIZE = 11
VOLCANO_TICK_LABEL_SIZE = 9.5
VOLCANO_TITLE_SIZE = 10

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    for path in [
        SUMMARY_DIR,
        TABLE_DIR / "rnaseq_volcano",
        TABLE_DIR / "rnaseq_counts",
        FIGURE_DIR / "flow",
        FIGURE_DIR / "rnaseq_volcano",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_manifest(hash_large_files: bool = False, large_file_limit: int = 1024**3) -> list[dict]:
    include = [
        ROOT / "table.tsv",
        ROOT / "A2_MFI_by_Gene.svg",
        FLOW_DIR,
        RNA_DIR / "JTRWM7_fastq.zip",
        RNA_DIR / "JTRWM7_bam",
        RNA_DIR / "annotation",
        RNA_DIR / "gene_count_cache",
        RNA_DIR / "JTRWM7_gene_counts_by_index.tsv",
        RNA_DIR / "JTRWM7_gene_count_summary.tsv",
        RNA_DIR / "volcano_results",
        RNA_DIR / "volcano_plots",
    ]
    rows = []
    skip_names = {".DS_Store"}
    skip_parts = {".matplotlib", "__pycache__"}
    for base in include:
        if not base.exists():
            continue
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for path in files:
            rel_parts = set(path.relative_to(ROOT).parts)
            if path.name in skip_names or rel_parts & skip_parts:
                continue
            size = path.stat().st_size
            row = {
                "relative_path": relpath(path),
                "size_bytes": int(size),
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
            if size > large_file_limit and not hash_large_files:
                row["sha256"] = ""
                row["sha256_note"] = "not_hashed_by_default_large_file"
            else:
                row["sha256"] = sha256_file(path)
                row["sha256_note"] = ""
            rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY_DIR / "raw_data_manifest.csv", index=False)
    return rows


def write_output_manifest() -> list[dict]:
    rows = []
    for base in [TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]:
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path.name in {"run_summary.json", ".DS_Store"}:
                continue
            rows.append(
                {
                    "relative_path": relpath(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return rows


def pdf_has_type3(path: Path) -> bool:
    data = path.read_bytes()
    return b"/Subtype /Type3" in data or b"/FontType 3" in data


def make_sample_map(bam_dir: Path = BAM_DIR, table_path: Path = TABLE_PATH) -> pd.DataFrame:
    table = pd.read_csv(table_path, sep="\t")
    missing = {"Order", "Index", "Gene", "No"} - set(table.columns)
    if missing:
        raise ValueError(f"table.tsv is missing required columns: {sorted(missing)}")

    bam_rows = []
    pattern = re.compile(r"JTRWM7_(\d+)_.*_dedup-mapped-reads\.bam$")
    for bam_path in sorted(bam_dir.glob("JTRWM7_*_dedup-mapped-reads.bam")):
        match = pattern.search(bam_path.name)
        if match:
            bam_rows.append({"file_order": int(match.group(1)), "bam_abs": bam_path})
    if not bam_rows:
        raise FileNotFoundError(f"No dedup-mapped BAM files found in {bam_dir}")

    bam_df = pd.DataFrame(bam_rows)
    mapped = bam_df.merge(table, left_on="file_order", right_on="Order", how="left", validate="one_to_one")
    if mapped["Index"].isna().any():
        missing_orders = mapped.loc[mapped["Index"].isna(), "file_order"].tolist()
        raise ValueError(f"BAM file numbers without table.tsv Order matches: {missing_orders}")

    mapped["Index"] = mapped["Index"].astype(int)
    mapped["Order"] = mapped["Order"].astype(int)
    mapped["No"] = mapped["No"].astype(str)
    mapped["sample_label"] = mapped["Gene"].astype(str) + "_" + mapped["No"]
    mapped["condition_label"] = "Index " + mapped["Index"].astype(str) + " " + mapped["sample_label"]
    mapped["bam_file"] = mapped["bam_abs"].map(lambda p: p.name)
    mapped["bam_path"] = mapped["bam_abs"].map(lambda p: relpath(p))
    mapped = mapped.sort_values("Index").reset_index(drop=True)

    expected = set(range(1, 13))
    observed = set(mapped["Index"].astype(int).tolist())
    if not expected.issubset(observed):
        raise ValueError(f"Expected mapped Index values 1..12; observed {sorted(observed)}")
    return mapped


def write_sample_map(sample_map: pd.DataFrame) -> Path:
    out = RNA_DIR / "JTRWM7_sample_map.tsv"
    cols = [
        "file_order",
        "bam_path",
        "Order",
        "Index",
        "Gene",
        "No",
        "gRNA",
        "Name",
        "Oligo",
        "Sample",
        "sample_label",
        "condition_label",
        "bam_file",
    ]
    sample_map[cols].to_csv(out, sep="\t", index=False)
    sample_map[cols].to_csv(TABLE_DIR / "rnaseq_counts" / "JTRWM7_sample_map.tsv", sep="\t", index=False)
    return out


def render_mfi_plot() -> dict:
    mfi = pd.read_csv(FLOW_DIR / "A2_MFI.csv")
    table = pd.read_csv(TABLE_PATH, sep="\t")
    sample_map = table[["Sample", "Gene", "Index"]].drop_duplicates("Sample")

    plot_df = (
        mfi[["Sample", "A2_MFI"]]
        .merge(sample_map, on="Sample", how="left", validate="many_to_one")
        .assign(_mfi_order=lambda d: np.arange(len(d)))
        .sort_values(["Index", "_mfi_order"], kind="mergesort", na_position="last")
        .drop(columns=["_mfi_order"])
        .reset_index(drop=True)
    )
    table_out = TABLE_DIR / "A2_MFI_by_Gene_table.csv"
    plot_df.to_csv(table_out, index=False)

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

    pdf_out = FIGURE_DIR / "flow" / "A2_MFI_by_Gene.pdf"
    png_out = FIGURE_DIR / "flow" / "A2_MFI_by_Gene.png"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, bbox_inches="tight", dpi=600)
    plt.close(fig)

    return {
        "table": relpath(table_out),
        "pdf": relpath(pdf_out),
        "png": relpath(png_out),
        "rows": int(len(plot_df)),
    }


def parse_color(value: str | None, gradients: dict[str, str]) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() == "none":
        return None
    match = re.match(r"url\(#([^)]+)\)", value)
    if match:
        return gradients.get(match.group(1), "#cccccc")
    return value


def parse_transform(text: str | None) -> np.ndarray:
    matrix = np.eye(3)
    if not text:
        return matrix
    for name, args in re.findall(r"(matrix|translate|scale)\(([^)]*)\)", text):
        nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", args)]
        if name == "matrix" and len(nums) == 6:
            a, b, c, d, e, f = nums
            local = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        elif name == "translate":
            tx = nums[0] if nums else 0.0
            ty = nums[1] if len(nums) > 1 else 0.0
            local = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        elif name == "scale":
            sx = nums[0] if nums else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            local = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
        else:
            continue
        matrix = matrix @ local
    return matrix


def transform_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    out = matrix @ np.array([x, y, 1.0])
    return float(out[0]), float(out[1])


def parse_path_d(d: str, matrix: np.ndarray) -> tuple[list[tuple[float, float]], list[int]]:
    tokens = re.findall(r"[MmLlZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    i = 0
    cmd = None
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    while i < len(tokens):
        token = tokens[i]
        if re.match(r"[MmLlZz]", token):
            cmd = token
            i += 1
        if cmd is None:
            break
        if cmd in {"M", "m", "L", "l"}:
            first = cmd in {"M", "m"}
            while i + 1 < len(tokens) and not re.match(r"[MmLlZz]", tokens[i]):
                x = float(tokens[i])
                y = float(tokens[i + 1])
                i += 2
                if cmd in {"m", "l"}:
                    x += current[0]
                    y += current[1]
                code = MplPath.MOVETO if first else MplPath.LINETO
                vertices.append(transform_point(matrix, x, y))
                codes.append(code)
                current = (x, y)
                if first:
                    start = current
                    first = False
                    if cmd == "M":
                        cmd = "L"
                    elif cmd == "m":
                        cmd = "l"
        elif cmd in {"Z", "z"}:
            vertices.append(transform_point(matrix, start[0], start[1]))
            codes.append(MplPath.CLOSEPOLY)
            current = start
            cmd = None
        else:
            i += 1
    return vertices, codes


def collect_gradients(root: ET.Element) -> dict[str, str]:
    gradients = {}
    for gradient in root.iter():
        if not gradient.tag.endswith("linearGradient"):
            continue
        gid = gradient.attrib.get("id")
        colors = [stop.attrib.get("stop-color") for stop in list(gradient) if stop.attrib.get("stop-color")]
        if gid and colors:
            gradients[gid] = colors[0]
    return gradients


def render_svg_subset(svg_path: Path, pdf_out: Path, png_out: Path) -> dict:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width = float(root.attrib.get("width", "578px").replace("px", ""))
    height = float(root.attrib.get("height", "688px").replace("px", ""))
    gradients = collect_gradients(root)
    unit_to_points = 72.0 / 100.0

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=300)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    path_count = 0
    text_count = 0

    def walk(element: ET.Element, inherited_style: dict, inherited_matrix: np.ndarray, in_defs: bool = False) -> None:
        nonlocal path_count, text_count
        tag = element.tag.split("}")[-1]
        if tag in {"defs", "font", "glyph", "missing-glyph", "clipPath", "linearGradient"}:
            in_defs = True

        style = inherited_style.copy()
        if "style" in element.attrib:
            for part in element.attrib["style"].split(";"):
                if ":" in part:
                    key, value = part.split(":", 1)
                    style[key.strip()] = value.strip()
        for key in [
            "fill",
            "stroke",
            "stroke-width",
            "stroke-opacity",
            "fill-opacity",
            "font-size",
            "font-weight",
            "font-family",
        ]:
            if key in element.attrib:
                style[key] = element.attrib[key]

        matrix = inherited_matrix @ parse_transform(element.attrib.get("transform"))

        if not in_defs and tag == "path" and "d" in element.attrib:
            vertices, codes = parse_path_d(element.attrib["d"], matrix)
            if vertices and codes:
                fill = parse_color(style.get("fill"), gradients)
                stroke = parse_color(style.get("stroke"), gradients)
                linewidth = float(style.get("stroke-width", 1) or 1) * unit_to_points
                alpha = float(style.get("fill-opacity", style.get("stroke-opacity", 1)) or 1)
                patch = PathPatch(
                    MplPath(vertices, codes),
                    facecolor=fill if fill else "none",
                    edgecolor=stroke if stroke else "none",
                    linewidth=linewidth,
                    alpha=alpha,
                    joinstyle="miter",
                    capstyle="projecting",
                )
                ax.add_patch(patch)
                path_count += 1

        if not in_defs and tag == "text":
            text = "".join(element.itertext()).rstrip("\n")
            if text:
                x = float(element.attrib.get("x", 0))
                y = float(element.attrib.get("y", 0))
                tx, ty = transform_point(matrix, x, y)
                rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
                font_size = float(style.get("font-size", 9) or 9) * unit_to_points
                font_weight = style.get("font-weight", "normal")
                color = parse_color(style.get("fill", "#000000"), gradients) or "#000000"
                ax.text(
                    tx,
                    ty,
                    text,
                    fontsize=font_size,
                    fontweight=font_weight,
                    color=color,
                    rotation=rotation,
                    rotation_mode="anchor",
                    va="baseline",
                    ha="left",
                    family="DejaVu Sans",
                )
                text_count += 1

        for child in list(element):
            walk(child, style, matrix, in_defs)

    walk(root, {}, np.eye(3))
    fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0)
    fig.savefig(png_out, bbox_inches="tight", pad_inches=0, dpi=600)
    plt.close(fig)

    svg_out = pdf_out.with_suffix(".svg")
    shutil.copy2(svg_path, svg_out)
    return {
        "source_svg": relpath(svg_path),
        "svg": relpath(svg_out),
        "pdf": relpath(pdf_out),
        "png": relpath(png_out),
        "paths_rendered": int(path_count),
        "text_labels_rendered": int(text_count),
    }


def render_flowjo_histogram() -> dict:
    return render_svg_subset(
        FLOW_DIR / "Histogram.svg",
        FIGURE_DIR / "flow" / "Histogram.pdf",
        FIGURE_DIR / "flow" / "Histogram.png",
    )


def validate_gtf(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing packaged GTF: {path}")
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            if len(line.rstrip("\n").split("\t")) == 9:
                return
            break
    raise ValueError(f"Packaged annotation does not look like a gzipped GTF: {path}")


def parse_gtf_attributes(attr_text: str) -> dict[str, str]:
    attrs = {}
    for part in attr_text.rstrip(";").split(";"):
        part = part.strip()
        if not part or " " not in part:
            continue
        key, value = part.split(" ", 1)
        attrs[key] = value.strip().strip('"')
    return attrs


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def load_exon_bins(gtf_path: Path = GTF_PATH, bin_size: int = BIN_SIZE):
    validate_gtf(gtf_path)
    gene_meta = {}
    intervals_by_gene = defaultdict(list)
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "exon":
                continue
            chrom, _, _, start, end, _, strand, _, attrs_text = fields
            attrs = parse_gtf_attributes(attrs_text)
            gene_id = attrs.get("gene_id")
            if not gene_id:
                continue
            intervals_by_gene[(chrom, gene_id)].append((int(start) - 1, int(end)))
            gene_meta[gene_id] = {
                "gene_id": gene_id,
                "gene_name": attrs.get("gene_name", gene_id),
                "gene_biotype": attrs.get("gene_biotype", attrs.get("gene_type", "")),
                "chrom": chrom,
                "strand": strand,
            }

    bins_by_chrom = defaultdict(lambda: defaultdict(list))
    for (chrom, gene_id), intervals in intervals_by_gene.items():
        for start, end in merge_intervals(intervals):
            for bin_id in range(start // bin_size, (end - 1) // bin_size + 1):
                bins_by_chrom[chrom][bin_id].append((start, end, gene_id))

    gene_meta_df = pd.DataFrame.from_dict(gene_meta, orient="index").sort_values(["chrom", "gene_name", "gene_id"])
    return bins_by_chrom, gene_meta_df


REF_CONSUMING_OPS = {0, 2, 3, 7, 8}
MATCH_OPS = {0, 7, 8}
SKIP_FLAG_MASK = 0x4 | 0x100 | 0x800 | 0x200
if SKIP_DUPLICATES:
    SKIP_FLAG_MASK |= 0x400


def read_bam_header(handle):
    magic = handle.read(4)
    if magic != b"BAM\x01":
        raise ValueError("Input is not a BAM file")
    header_len = struct.unpack("<i", handle.read(4))[0]
    header_text = handle.read(header_len).decode(errors="replace")
    n_ref = struct.unpack("<i", handle.read(4))[0]
    refs = []
    for _ in range(n_ref):
        name_len = struct.unpack("<i", handle.read(4))[0]
        name = handle.read(name_len)[:-1].decode(errors="replace")
        ref_len = struct.unpack("<i", handle.read(4))[0]
        refs.append((name, ref_len))
    return header_text, refs


def parse_nh_tag(aux: bytes):
    i = 0
    n = len(aux)
    while i + 3 <= n:
        tag = aux[i : i + 2]
        typ = chr(aux[i + 2])
        i += 3
        if typ == "A":
            value = aux[i]
            i += 1
        elif typ == "c":
            value = struct.unpack("<b", aux[i : i + 1])[0]
            i += 1
        elif typ == "C":
            value = aux[i]
            i += 1
        elif typ == "s":
            value = struct.unpack("<h", aux[i : i + 2])[0]
            i += 2
        elif typ == "S":
            value = struct.unpack("<H", aux[i : i + 2])[0]
            i += 2
        elif typ == "i":
            value = struct.unpack("<i", aux[i : i + 4])[0]
            i += 4
        elif typ == "I":
            value = struct.unpack("<I", aux[i : i + 4])[0]
            i += 4
        elif typ == "f":
            value = struct.unpack("<f", aux[i : i + 4])[0]
            i += 4
        elif typ in {"Z", "H"}:
            j = aux.index(b"\0", i)
            value = aux[i:j].decode(errors="replace")
            i = j + 1
        elif typ == "B":
            subtype = chr(aux[i])
            count = struct.unpack("<i", aux[i + 1 : i + 5])[0]
            i += 5
            sizes = {"c": 1, "C": 1, "s": 2, "S": 2, "i": 4, "I": 4, "f": 4}
            i += sizes[subtype] * count
            value = None
        else:
            return None
        if tag == b"NH":
            return value
    return None


def cigar_blocks(pos: int, cigar_values) -> list[tuple[int, int]]:
    ref_pos = pos
    blocks = []
    for value in cigar_values:
        op = value & 0xF
        length = value >> 4
        if op in MATCH_OPS:
            blocks.append((ref_pos, ref_pos + length))
        if op in REF_CONSUMING_OPS:
            ref_pos += length
    return blocks


def genes_for_blocks(chrom: str, blocks: list[tuple[int, int]], bins_by_chrom, bin_size: int = BIN_SIZE) -> set[str]:
    chrom_bins = bins_by_chrom.get(chrom)
    if not chrom_bins:
        return set()
    genes = set()
    seen = set()
    for block_start, block_end in blocks:
        if block_end <= block_start:
            continue
        for bin_id in range(block_start // bin_size, (block_end - 1) // bin_size + 1):
            for exon_start, exon_end, gene_id in chrom_bins.get(bin_id, []):
                key = (exon_start, exon_end, gene_id)
                if key in seen:
                    continue
                seen.add(key)
                if exon_start < block_end and exon_end > block_start:
                    genes.add(gene_id)
    return genes


def count_one_bam(bam_path: Path, bins_by_chrom, progress_every: int = 5_000_000):
    counts = Counter()
    summary = Counter()
    with gzip.open(bam_path, "rb") as handle:
        _, refs = read_bam_header(handle)
        ref_names = [name for name, _ in refs]
        while True:
            raw_size = handle.read(4)
            if not raw_size:
                break
            block_size = struct.unpack("<i", raw_size)[0]
            record = handle.read(block_size)
            summary["records"] += 1

            ref_id, pos = struct.unpack("<ii", record[0:8])
            read_name_len = record[8]
            n_cigar = struct.unpack("<H", record[12:14])[0]
            flag = struct.unpack("<H", record[14:16])[0]
            seq_len = struct.unpack("<i", record[16:20])[0]

            if flag & SKIP_FLAG_MASK or ref_id < 0:
                summary["skipped_flag_or_unmapped"] += 1
                continue

            offset = 32 + read_name_len
            cigar_values = struct.unpack(f"<{n_cigar}I", record[offset : offset + 4 * n_cigar]) if n_cigar else []
            offset += 4 * n_cigar + ((seq_len + 1) // 2) + seq_len
            aux = record[offset:]

            if not COUNT_MULTIMAPPERS:
                nh = parse_nh_tag(aux)
                if nh is not None and nh > 1:
                    summary["skipped_multimapper"] += 1
                    continue

            blocks = cigar_blocks(pos, cigar_values)
            genes = genes_for_blocks(ref_names[ref_id], blocks, bins_by_chrom)
            if len(genes) == 1:
                counts[next(iter(genes))] += 1
                summary["counted"] += 1
            elif len(genes) == 0:
                summary["no_feature"] += 1
            else:
                summary["ambiguous"] += 1

            if progress_every and summary["records"] % progress_every == 0:
                print(f"{bam_path.name}: {summary['records']:,} records, {summary['counted']:,} counted")
    return counts, summary


def sample_cache_paths(row) -> tuple[Path, Path]:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", row.sample_label)
    stem = f"Index_{int(row.Index):02d}_{safe_label}"
    return COUNT_DIR / f"{stem}.counts.tsv", COUNT_DIR / f"{stem}.summary.tsv"


def count_or_load_sample(row, bins_by_chrom, gene_meta: pd.DataFrame, force_recount_bam: bool = False):
    count_path, summary_path = sample_cache_paths(row)
    if count_path.exists() and not force_recount_bam:
        counts = pd.read_csv(count_path, sep="\t")
        summary = pd.read_csv(summary_path, sep="\t") if summary_path.exists() else pd.DataFrame()
        return counts, summary

    print(f"Counting {row.condition_label}: {Path(row.bam_abs).name}")
    counts_counter, summary_counter = count_one_bam(Path(row.bam_abs), bins_by_chrom)
    counts = pd.DataFrame({"gene_id": list(counts_counter.keys()), "count": list(counts_counter.values())})
    counts = gene_meta[["gene_id", "gene_name", "gene_biotype", "chrom", "strand"]].merge(counts, on="gene_id", how="left")
    counts["count"] = counts["count"].fillna(0).astype(int)
    counts.to_csv(count_path, sep="\t", index=False)

    summary = pd.DataFrame(sorted(summary_counter.items()), columns=["metric", "value"])
    summary.to_csv(summary_path, sep="\t", index=False)
    return counts, summary


def normalize_count_matrix_columns(count_matrix: pd.DataFrame) -> pd.DataFrame:
    rename = {str(i): i for i in range(1, 13) if str(i) in count_matrix.columns}
    count_matrix = count_matrix.rename(columns=rename)
    sample_indices = [i for i in range(1, 13) if i in count_matrix.columns]
    count_matrix[sample_indices] = count_matrix[sample_indices].fillna(0).astype(int)
    return count_matrix


def build_count_matrix(sample_map: pd.DataFrame, force_recount_bam: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins_by_chrom, gene_meta = load_exon_bins()
    count_tables = []
    summary_tables = []
    for row in sample_map.itertuples(index=False):
        counts, summary = count_or_load_sample(row, bins_by_chrom, gene_meta, force_recount_bam=force_recount_bam)
        counts = counts.rename(columns={"count": int(row.Index)})
        count_tables.append(counts[["gene_id", int(row.Index)]])
        if not summary.empty:
            summary = summary.copy()
            summary["Index"] = int(row.Index)
            summary["sample_label"] = row.sample_label
            summary_tables.append(summary)

    count_matrix = gene_meta[["gene_id", "gene_name", "gene_biotype", "chrom", "strand"]].copy()
    for table_counts in count_tables:
        count_matrix = count_matrix.merge(table_counts, on="gene_id", how="left")
    count_matrix = normalize_count_matrix_columns(count_matrix)
    count_summary = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    return count_matrix, count_summary


def load_or_build_count_matrix(sample_map: pd.DataFrame, rebuild_count_matrix: bool, force_recount_bam: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if rebuild_count_matrix or force_recount_bam or not COUNT_MATRIX_PATH.exists():
        count_matrix, count_summary = build_count_matrix(sample_map, force_recount_bam=force_recount_bam)
        mode = "rebuilt_from_per_sample_count_cache_or_bam"
    else:
        count_matrix = pd.read_csv(COUNT_MATRIX_PATH, sep="\t", low_memory=False)
        count_summary_path = RNA_DIR / "JTRWM7_gene_count_summary.tsv"
        count_summary = pd.read_csv(count_summary_path, sep="\t") if count_summary_path.exists() else pd.DataFrame()
        mode = "loaded_packaged_count_matrix"
    count_matrix = normalize_count_matrix_columns(count_matrix)
    count_matrix.to_csv(TABLE_DIR / "rnaseq_counts" / "JTRWM7_gene_counts_by_index.tsv", sep="\t", index=False)
    if not count_summary.empty:
        count_summary.to_csv(TABLE_DIR / "rnaseq_counts" / "JTRWM7_gene_count_summary.tsv", sep="\t", index=False)
    return count_matrix, count_summary, mode


def bh_adjust(pvalues) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full_like(pvalues, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return qvalues
    p = pvalues[finite]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(p)
    out[order] = adjusted
    qvalues[finite] = out
    return qvalues


def label_for_indices(indices, sample_map: pd.DataFrame) -> str:
    by_index = sample_map.set_index("Index")
    pieces = []
    for idx in indices:
        row = by_index.loc[int(idx)]
        pieces.append(f"Index {int(idx)} {row['sample_label']}")
    return " + ".join(pieces)


def safe_name(text: str) -> str:
    text = text.replace("+", "plus")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def volcano_table(count_matrix: pd.DataFrame, sample_map: pd.DataFrame, reference_indices, test_indices, comparison_name: str) -> pd.DataFrame:
    reference_indices = [int(i) for i in reference_indices]
    test_indices = [int(i) for i in test_indices]

    ref_counts = count_matrix[reference_indices].sum(axis=1).astype(int)
    test_counts = count_matrix[test_indices].sum(axis=1).astype(int)
    ref_lib = int(ref_counts.sum())
    test_lib = int(test_counts.sum())
    total_counts = ref_counts + test_counts

    keep = total_counts >= MIN_TOTAL_COUNTS
    df = count_matrix.loc[keep, ["gene_id", "gene_name", "gene_biotype", "chrom", "strand"]].copy()
    df["reference_count"] = ref_counts.loc[keep].to_numpy()
    df["test_count"] = test_counts.loc[keep].to_numpy()
    df["reference_cpm"] = df["reference_count"] / ref_lib * 1_000_000
    df["test_cpm"] = df["test_count"] / test_lib * 1_000_000
    df["log2_fc"] = np.log2((df["test_cpm"] + PRIOR_CPM) / (df["reference_cpm"] + PRIOR_CPM))

    pvalues = []
    for test_gene, ref_gene in zip(df["test_count"].to_numpy(), df["reference_count"].to_numpy()):
        table = [[test_gene, test_lib - test_gene], [ref_gene, ref_lib - ref_gene]]
        pvalues.append(fisher_exact(table, alternative="two-sided").pvalue)

    df["pvalue"] = pvalues
    df["fdr"] = bh_adjust(df["pvalue"])
    df["neg_log10_fdr"] = -np.log10(df["fdr"].clip(lower=1e-300))
    df["significant"] = (df["fdr"] < FDR_THRESHOLD) & (df["log2_fc"].abs() >= LFC_THRESHOLD)
    df["comparison"] = comparison_name
    df["reference"] = label_for_indices(reference_indices, sample_map)
    df["test"] = label_for_indices(test_indices, sample_map)
    return df.sort_values(["fdr", "pvalue", "log2_fc"], ascending=[True, True, False])


def extreme_lfc_label_genes(df: pd.DataFrame, per_side: int = TOP_LFC_LABELS_PER_SIDE) -> pd.DataFrame:
    finite = df[np.isfinite(df["log2_fc"]) & np.isfinite(df["neg_log10_fdr"])].copy()
    if finite.empty or per_side <= 0:
        return finite.iloc[0:0].copy()
    lowest = finite.nsmallest(per_side, "log2_fc")
    highest = finite.nlargest(per_side, "log2_fc")
    labels = pd.concat([lowest, highest], ignore_index=False).drop_duplicates("gene_id", keep="first")
    return labels.sort_values("log2_fc")


def custom_label_genes(df: pd.DataFrame, genes, warn_missing: bool = True) -> pd.DataFrame:
    if genes is None:
        return df.iloc[0:0].copy()
    if isinstance(genes, str):
        genes = [genes]

    requested = {}
    for gene in genes:
        if gene is None:
            continue
        try:
            if pd.isna(gene):
                continue
        except (TypeError, ValueError):
            pass
        gene = str(gene).strip()
        if gene:
            requested[gene.casefold()] = gene
    if not requested:
        return df.iloc[0:0].copy()

    finite = np.isfinite(df["log2_fc"]) & np.isfinite(df["neg_log10_fdr"])
    gene_names = df["gene_name"].fillna("").astype(str).str.strip().str.casefold()
    gene_ids = df["gene_id"].fillna("").astype(str).str.strip().str.casefold()
    gene_ids_without_version = gene_ids.str.replace(r"\.\d+$", "", regex=True)
    requested_normalized = set(requested)
    custom_mask = finite & (
        gene_names.isin(requested_normalized)
        | gene_ids.isin(requested_normalized)
        | gene_ids_without_version.isin(requested_normalized)
    )
    selected = df.loc[custom_mask].copy()

    if warn_missing:
        matched_identifiers = (
            set(gene_names.loc[custom_mask])
            | set(gene_ids.loc[custom_mask])
            | set(gene_ids_without_version.loc[custom_mask])
        )
        missing = [original for normalized, original in requested.items() if normalized not in matched_identifiers]
        if missing:
            warnings.warn(
                "Custom volcano labels not found after count filtering: " + ", ".join(missing),
                stacklevel=2,
            )
    return selected


def gene_label_text(row: pd.Series) -> str:
    gene_name = row.get("gene_name")
    if gene_name is not None and not pd.isna(gene_name) and str(gene_name).strip():
        return str(gene_name).strip()
    gene_id = row.get("gene_id", "")
    return "" if pd.isna(gene_id) else str(gene_id).strip()


def select_volcano_label_genes(
    df: pd.DataFrame,
    custom_genes=None,
    per_side: int = TOP_LFC_LABELS_PER_SIDE,
) -> pd.DataFrame:
    custom = custom_label_genes(df, genes=custom_genes, warn_missing=True).assign(label_source="custom")
    extreme = extreme_lfc_label_genes(df, per_side=per_side).assign(label_source="extreme_lfc")
    labels = pd.concat([custom, extreme], ignore_index=False)
    if labels.empty:
        labels["display_label"] = pd.Series(dtype=str)
        return labels

    labels = labels.copy()
    labels["_label_dedup_key"] = labels["gene_id"].fillna("").astype(str).str.strip()
    missing_id = labels["_label_dedup_key"].eq("")
    labels.loc[missing_id, "_label_dedup_key"] = "__row_" + labels.index[missing_id].astype(str)
    labels = labels.drop_duplicates("_label_dedup_key", keep="first").drop(columns="_label_dedup_key")
    labels["display_label"] = labels.apply(gene_label_text, axis=1)
    return labels.sort_values("log2_fc")


def add_repel_labels(ax, label_df: pd.DataFrame) -> None:
    texts = []
    target_x = []
    target_y = []
    for _, row in label_df.iterrows():
        x = float(row["log2_fc"])
        y = float(row["neg_log10_fdr"])
        ha = "right" if x < 0 else "left"
        label = row.get("display_label", gene_label_text(row))
        if not label:
            continue
        texts.append(ax.text(x, y, label, fontsize=VOLCANO_GENE_LABEL_SIZE, ha=ha, va="center", color=VOLCANO_AXIS_COLOR))
        target_x.append(x)
        target_y.append(y)

    if not texts:
        return

    random_state = np.random.get_state()
    np.random.seed(VOLCANO_LABEL_LAYOUT_SEED)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(
                texts,
                x=target_x,
                y=target_y,
                target_x=target_x,
                target_y=target_y,
                ax=ax,
                expand=(1.2, 1.45),
                force_text=(0.25, 0.5),
                force_static=(0.08, 0.18),
                force_pull=(0.01, 0.02),
                arrowprops={"arrowstyle": "-", "color": VOLCANO_GUIDE_COLOR, "linewidth": 0.6, "alpha": 0.75},
                min_arrow_len=3,
                iter_lim=300,
            )
    finally:
        np.random.set_state(random_state)


def plot_volcano_from_table(
    df: pd.DataFrame,
    label_df: pd.DataFrame,
    sample_map: pd.DataFrame,
    comparison_name: str,
    reference_indices,
    test_indices,
) -> dict:
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=300)
    sig_up = df["significant"] & (df["log2_fc"] > 0)
    sig_down = df["significant"] & (df["log2_fc"] < 0)
    nonsig = ~(sig_up | sig_down)

    ax.scatter(df.loc[nonsig, "log2_fc"], df.loc[nonsig, "neg_log10_fdr"], s=VOLCANO_NONSIG_POINT_SIZE, c=VOLCANO_NEUTRAL_COLOR, alpha=0.36, linewidths=0, rasterized=True)
    ax.scatter(df.loc[sig_down, "log2_fc"], df.loc[sig_down, "neg_log10_fdr"], s=VOLCANO_SIG_POINT_SIZE, c=VOLCANO_DOWN_COLOR, alpha=0.92, linewidths=0, rasterized=True)
    ax.scatter(df.loc[sig_up, "log2_fc"], df.loc[sig_up, "neg_log10_fdr"], s=VOLCANO_SIG_POINT_SIZE, c=VOLCANO_UP_COLOR, alpha=0.92, linewidths=0, rasterized=True)

    ax.axvline(-LFC_THRESHOLD, color=VOLCANO_GUIDE_COLOR, linestyle="--", linewidth=0.9)
    ax.axvline(LFC_THRESHOLD, color=VOLCANO_GUIDE_COLOR, linestyle="--", linewidth=0.9)
    ax.axhline(-math.log10(FDR_THRESHOLD), color=VOLCANO_GUIDE_COLOR, linestyle="--", linewidth=0.9)

    add_repel_labels(ax, label_df)

    ref_label = label_for_indices(reference_indices, sample_map)
    test_label = label_for_indices(test_indices, sample_map)
    ax.set_title(f"{comparison_name}: {test_label} vs {ref_label}", fontsize=VOLCANO_TITLE_SIZE, color=VOLCANO_AXIS_COLOR, pad=10)
    ax.set_xlabel("log2 fold change (test / reference)", fontsize=VOLCANO_AXIS_LABEL_SIZE, color=VOLCANO_AXIS_COLOR)
    ax.set_ylabel("-log10 FDR", fontsize=VOLCANO_AXIS_LABEL_SIZE, color=VOLCANO_AXIS_COLOR)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(VOLCANO_AXIS_COLOR)
    ax.spines[["left", "bottom"]].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=VOLCANO_TICK_LABEL_SIZE, colors=VOLCANO_AXIS_COLOR, width=0.9, length=3.5)
    ax.grid(True, axis="y", color=VOLCANO_GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_stem = safe_name(comparison_name)
    pdf_path = FIGURE_DIR / "rnaseq_volcano" / f"{out_stem}.pdf"
    png_path = FIGURE_DIR / "rnaseq_volcano" / f"{out_stem}.png"
    svg_path = FIGURE_DIR / "rnaseq_volcano" / f"{out_stem}.svg"
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": relpath(pdf_path), "png": relpath(png_path), "svg": relpath(svg_path)}


def compare_generated_to_packaged(generated: pd.DataFrame, packaged_path: Path) -> dict:
    if not packaged_path.exists():
        return {"packaged_table": relpath(packaged_path), "status": "missing"}
    packaged = pd.read_csv(packaged_path, sep="\t", low_memory=False)
    result = {
        "packaged_table": relpath(packaged_path),
        "status": "compared",
        "generated_rows": int(len(generated)),
        "packaged_rows": int(len(packaged)),
    }
    common = [col for col in generated.columns if col in packaged.columns]
    bool_cols = [col for col in common if pd.api.types.is_bool_dtype(generated[col]) and pd.api.types.is_bool_dtype(packaged[col])]
    numeric_cols = [
        col
        for col in common
        if col not in bool_cols and pd.api.types.is_numeric_dtype(generated[col]) and pd.api.types.is_numeric_dtype(packaged[col])
    ]
    max_abs_diff = 0.0
    for col in numeric_cols:
        left = pd.to_numeric(generated[col], errors="coerce").to_numpy()
        right = pd.to_numeric(packaged[col], errors="coerce").to_numpy()
        if len(left) != len(right):
            max_abs_diff = float("nan")
            break
        diff = np.nanmax(np.abs(left - right)) if len(left) else 0.0
        max_abs_diff = max(max_abs_diff, float(diff))
    result["numeric_columns_compared"] = numeric_cols
    result["max_abs_numeric_diff"] = max_abs_diff
    result["boolean_columns_compared"] = bool_cols
    for col in bool_cols:
        if len(generated[col]) == len(packaged[col]):
            result[f"{col}_matches"] = bool(generated[col].astype(bool).tolist() == packaged[col].astype(bool).tolist())
    if "gene_id" in common:
        result["gene_id_order_matches"] = bool(generated["gene_id"].astype(str).tolist() == packaged["gene_id"].astype(str).tolist())
    return result


def run_rnaseq_volcano(sample_map: pd.DataFrame, count_matrix: pd.DataFrame) -> dict:
    comparisons = [
        {"name": "Index_4_vs_3", "reference_indices": [4], "test_indices": [3], "custom_label_genes": []},
        {"name": "Index_6_vs_5", "reference_indices": [6], "test_indices": [5], "custom_label_genes": []},
        {"name": "Index_8_vs_7", "reference_indices": [8], "test_indices": [7], "custom_label_genes": []},
        {
            "name": "Index_10_vs_9",
            "reference_indices": [10],
            "test_indices": [9],
            "custom_label_genes": list(CUSTOM_LABEL_GENES_10_9),
        },
    ]
    all_results = []
    details = []
    for comparison in comparisons:
        df = volcano_table(
            count_matrix,
            sample_map,
            reference_indices=comparison["reference_indices"],
            test_indices=comparison["test_indices"],
            comparison_name=comparison["name"],
        )
        table_path = TABLE_DIR / "rnaseq_volcano" / f"{safe_name(comparison['name'])}.tsv"
        df.to_csv(table_path, sep="\t", index=False)
        label_df = select_volcano_label_genes(
            df,
            custom_genes=comparison["custom_label_genes"],
            per_side=TOP_LFC_LABELS_PER_SIDE,
        )
        label_path = TABLE_DIR / "rnaseq_volcano" / f"{safe_name(comparison['name'])}_labels.tsv"
        label_df.to_csv(label_path, sep="\t", index=False)
        figure_paths = plot_volcano_from_table(
            df,
            label_df,
            sample_map,
            comparison["name"],
            comparison["reference_indices"],
            comparison["test_indices"],
        )
        packaged_compare = compare_generated_to_packaged(df, RNA_DIR / "volcano_results" / f"{safe_name(comparison['name'])}.tsv")
        details.append(
            {
                "comparison": comparison["name"],
                "table": relpath(table_path),
                "label_table": relpath(label_path),
                "rows": int(len(df)),
                "significant_rows": int(df["significant"].sum()),
                "label_rows": int(len(label_df)),
                "custom_labels_requested": comparison["custom_label_genes"],
                "custom_labels_plotted": label_df.loc[label_df["label_source"].eq("custom"), "display_label"].tolist(),
                **figure_paths,
                "packaged_result_check": packaged_compare,
            }
        )
        all_results.append(df)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_path = TABLE_DIR / "rnaseq_volcano" / "all_pairwise_volcano_results.tsv"
    all_results_df.to_csv(all_results_path, sep="\t", index=False)
    return {"comparisons": details, "all_pairwise_table": relpath(all_results_path), "all_pairwise_rows": int(len(all_results_df))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate Extended Data Figure 8 audit tables, figures, and summaries.")
    parser.add_argument("--hash-large-files", action="store_true", help="Compute SHA256 hashes for files larger than 1 GiB.")
    parser.add_argument("--rebuild-count-matrix", action="store_true", help="Rebuild the RNA-seq count matrix from per-sample count caches, counting BAMs only if a cache is missing.")
    parser.add_argument("--force-recount-bam", action="store_true", help="Recount all packaged BAM files against the packaged GTF before rebuilding RNA-seq figures. This is slow.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    started_at = utc_now()
    ensure_dirs()

    raw_manifest = write_manifest(hash_large_files=args.hash_large_files)
    sample_map = make_sample_map()
    sample_map_path = write_sample_map(sample_map)

    flow_mfi = render_mfi_plot()
    flow_histogram = render_flowjo_histogram()

    count_matrix, count_summary, count_mode = load_or_build_count_matrix(
        sample_map,
        rebuild_count_matrix=args.rebuild_count_matrix,
        force_recount_bam=args.force_recount_bam,
    )
    rnaseq = run_rnaseq_volcano(sample_map, count_matrix)

    pdfs = sorted(FIGURE_DIR.rglob("*.pdf"))
    pngs = sorted(FIGURE_DIR.rglob("*.png"))
    svgs = sorted(FIGURE_DIR.rglob("*.svg"))
    qc_summary = {
        "generated_at_utc": utc_now(),
        "sample_map": {"path": relpath(sample_map_path), "rows": int(len(sample_map))},
        "flow_mfi": flow_mfi,
        "flow_histogram": flow_histogram,
        "rnaseq_count_matrix": {
            "mode": count_mode,
            "rows": int(len(count_matrix)),
            "columns": [str(c) for c in count_matrix.columns],
            "summary_rows": int(len(count_summary)) if count_summary is not None else 0,
        },
        "rnaseq_volcano": rnaseq,
        "volcano_label_layout": {
            "backend": "adjustText",
            "adjustText_version": adjustText.__version__,
            "random_seed": VOLCANO_LABEL_LAYOUT_SEED,
            "extreme_labels_per_side": TOP_LFC_LABELS_PER_SIDE,
        },
        "figures": {
            "pdf": [relpath(path) for path in pdfs],
            "png": [relpath(path) for path in pngs],
            "svg": [relpath(path) for path in svgs],
        },
        "pdf_font_check": {relpath(path): {"contains_type3_font": pdf_has_type3(path)} for path in pdfs},
        "packaged_inputs": {
            "manifest": relpath(SUMMARY_DIR / "raw_data_manifest.csv"),
            "manifest_rows": int(len(raw_manifest)),
            "large_files_hashed": bool(args.hash_large_files),
        },
        "notes": [
            "The FlowJo histogram is rendered from the packaged Histogram.svg exported with the WSP/FCS inputs; no FCS gating script was present in the source folder.",
            "The RNA-seq default run uses the packaged count matrix; --rebuild-count-matrix and --force-recount-bam are provided for upstream count regeneration from packaged BAM/GTF inputs.",
            "The four RNA-seq comparisons are declared from a fresh list on every run to prevent notebook-state accumulation and duplicate aggregate rows.",
            "Index_10_vs_9 adds CALR to the ten extreme-log2-fold-change labels; the exact plotted label rows are saved beside each volcano table.",
        ],
    }
    write_json(SUMMARY_DIR / "qc_summary.json", qc_summary)

    output_manifest = write_output_manifest()
    run_summary = {
        "workflow": "Extended Data Figure 8 audit workflow",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "duration_seconds": round(time.time() - start, 3),
        "arguments": vars(args),
        "matplotlib_rcparams": {
            "pdf.fonttype": plt.rcParams["pdf.fonttype"],
            "ps.fonttype": plt.rcParams["ps.fonttype"],
            "svg.fonttype": plt.rcParams["svg.fonttype"],
            "font.family": plt.rcParams["font.family"],
            "font.sans-serif": plt.rcParams["font.sans-serif"],
        },
        "outputs": output_manifest,
    }
    write_json(SUMMARY_DIR / "run_summary.json", run_summary)
    print(f"Wrote {len(output_manifest)} output files. QC summary: {SUMMARY_DIR / 'qc_summary.json'}")


if __name__ == "__main__":
    main()
