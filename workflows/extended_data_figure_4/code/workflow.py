#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/extended_data_fig4_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "outputs" / "summaries"
TABLE_DIR = ROOT / "data" / "figure_tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


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
        TABLE_DIR,
        FIGURE_DIR / "barcode_proportions",
        FIGURE_DIR / "flow_panels",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_manifest(hash_large_files: bool = False, large_file_limit: int = 1024**3) -> list[dict]:
    include = [
        ROOT / "BC.txt",
        ROOT / "PL47-D03",
        ROOT / "PL47-D04",
        ROOT / "PL47-D05",
        ROOT / "PL47-D05 Analysis",
        ROOT / "PL47-D06",
        ROOT / "PL47-D06 Analysis",
        ROOT / "PL47-D06 Figure Analysis",
        ROOT / "barcode_hit_table.csv",
    ]
    rows = []
    for base in include:
        if not base.exists():
            continue
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for path in files:
            if path.name == ".DS_Store" or path.name.startswith("~$"):
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
    out = SUMMARY_DIR / "raw_data_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return rows


def write_output_manifest() -> list[dict]:
    rows = []
    for base in [TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]:
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path.name in {"run_summary.json", ".DS_Store"}:
                continue
            rows.append({"relative_path": relpath(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def iter_fastq_sequences_gz(path: Path):
    with gzip.open(path, "rt", encoding="ascii", errors="ignore") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().strip()
            fh.readline()
            fh.readline()
            yield seq


def load_barcodes(path: Path) -> pd.DataFrame:
    bc = pd.read_csv(path, sep=None, engine="python")
    missing = {"Group", "BC"} - set(bc.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    bc = bc[["Group", "BC"]].dropna().copy()
    bc["Group"] = bc["Group"].astype(str).str.strip()
    bc["BC"] = bc["BC"].astype(str).str.upper().str.strip()
    return bc


def compile_group_regex(bc: pd.DataFrame, context_5: str = "", context_3: str = "", max_edits: int = 1):
    parts = []
    id_to_group = {}
    for i, row in bc.iterrows():
        pid = f"p{i}"
        motif = f"{context_5}{row['BC']}{context_3}"
        rc = reverse_complement(motif)
        parts.append(rf"(?P<{pid}>(?:{regex.escape(motif)}|{regex.escape(rc)}))" + rf"{{e<={max_edits}}}")
        id_to_group[pid] = row["Group"]
    pattern = "|".join(parts)
    return regex.compile(pattern, flags=regex.VERSION1 | regex.BESTMATCH | regex.IGNORECASE), id_to_group


def count_best_read_matches(fastq_dir: Path, bc_path: Path) -> pd.DataFrame:
    bc = load_barcodes(bc_path)
    groups = list(dict.fromkeys(bc["Group"].tolist()))
    rx, id_to_group = compile_group_regex(bc, context_5="GGAGT", context_3="TCGGC", max_edits=1)
    rows = []
    for fastq in sorted(fastq_dir.glob("*.fastq.gz")):
        counts = dict.fromkeys(groups, 0)
        total_reads = matched_reads = unassigned_reads = 0
        for seq in iter_fastq_sequences_gz(fastq):
            total_reads += 1
            match = rx.search(seq)
            if match:
                matched_reads += 1
                counts[id_to_group[match.lastgroup]] += 1
            else:
                unassigned_reads += 1
        total_hits = int(sum(counts.values()))
        row = {
            "fastq": fastq.name,
            "total_reads": int(total_reads),
            "matched_reads": int(matched_reads),
            "unassigned_reads": int(unassigned_reads),
            "total_hits": total_hits,
        }
        for group in groups:
            row[group] = int(counts[group])
        for group in groups:
            row[f"prop_{group}"] = counts[group] / total_hits if total_hits else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fastq").reset_index(drop=True)


def count_barcode_occurrences(fastq_dir: Path, bc_path: Path, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bc = load_barcodes(bc_path)
    groups = list(dict.fromkeys(bc["Group"].tolist()))
    rows = []
    count_matches = {}
    for fastq in sorted(fastq_dir.glob("*.fastq.gz")):
        count_rows = []
        for _, row in bc.iterrows():
            motif = row["BC"]
            rc = reverse_complement(motif)
            rx = regex.compile(rf"(?:{regex.escape(motif)}|{regex.escape(rc)})" + r"{e<=1}", flags=regex.VERSION1 | regex.BESTMATCH | regex.IGNORECASE)
            count = 0
            for seq in iter_fastq_sequences_gz(fastq):
                count += len(list(rx.finditer(seq, overlapped=False)))
            count_rows.append({"Group": row["Group"], "BC": row["BC"], "Count": int(count)})
        count_df = pd.DataFrame(count_rows)
        out_path = out_dir / f"Counts_{fastq.stem.replace('.fastq', '')}.txt"
        count_df.to_csv(out_path, sep="\t", index=False)
        totals = count_df.groupby("Group", sort=False)["Count"].sum().reindex(groups).fillna(0).astype(int)
        total = int(totals.sum())
        row = {"FastqFile": fastq.name}
        for group in groups:
            row[group] = int(totals[group])
        for group in groups:
            row[f"prop_{group}"] = totals[group] / total if total else 0.0
        rows.append(row)

        reference = fastq_dir / out_path.name
        if reference.exists():
            ref = pd.read_csv(reference, sep="\t")
            count_matches[out_path.name] = bool(ref[["Group", "BC", "Count"]].equals(count_df[["Group", "BC", "Count"]]))
    summary = pd.DataFrame(rows).sort_values("FastqFile").reset_index(drop=True)
    summary.to_csv(out_dir / "Summary_rebuilt.txt", sep="\t", index=False)
    return summary, count_matches


def build_d05_tables() -> dict:
    src = ROOT / "PL47-D05 Analysis" / "Summary_edit.txt"
    summary_edit = pd.read_csv(src, sep="\t")
    summary_edit.to_csv(TABLE_DIR / "PL47_D05_summary_edit.csv", index=False)
    rebuilt, count_matches = count_barcode_occurrences(ROOT / "PL47-D05 Analysis", ROOT / "PL47-D05 Analysis" / "BC.txt", TABLE_DIR / "PL47_D05_rebuilt_counts")
    rebuilt.to_csv(TABLE_DIR / "PL47_D05_rebuilt_summary.csv", index=False)
    return {
        "summary_edit_rows": int(len(summary_edit)),
        "rebuilt_summary_rows": int(len(rebuilt)),
        "counts_exact_match": bool(count_matches and all(count_matches.values())),
    }


def build_d06_tables() -> dict:
    table = count_best_read_matches(ROOT / "PL47-D06 Analysis", ROOT / "BC.txt")
    table.to_csv(TABLE_DIR / "PL47_D06_barcode_hit_table.csv", index=False)
    for label, fastqs in D06_FASTQ_GROUPS.items():
        subset = table[table["fastq"].isin(fastqs)].set_index("fastq").loc[fastqs].reset_index()
        subset.to_csv(TABLE_DIR / f"PL47_D06_{label}_proportions.csv", index=False)
    comparisons = {}
    for reference in [
        ROOT / "barcode_hit_table.csv",
        ROOT / "PL47-D06 Figure Analysis" / "barcode_hit_table.csv",
    ]:
        if reference.exists():
            ref = pd.read_csv(reference)
            comparisons[relpath(reference)] = {
                "shape_match": list(ref.shape) == list(table.shape),
                "exact_match": ref.equals(table),
                "max_numeric_abs_diff": max_numeric_abs_diff(ref, table),
            }
    return {"rows": int(len(table)), "comparisons": comparisons}


def max_numeric_abs_diff(a: pd.DataFrame, b: pd.DataFrame) -> float | None:
    common = [c for c in a.columns if c in b.columns]
    diffs = []
    for col in common:
        av = pd.to_numeric(a[col], errors="coerce")
        bv = pd.to_numeric(b[col], errors="coerce")
        mask = av.notna() & bv.notna()
        if mask.any():
            diffs.append(float(np.max(np.abs(av[mask].to_numpy() - bv[mask].to_numpy()))))
    return max(diffs) if diffs else None


D06_FASTQ_GROUPS = {
    "T_cells": [f"{i:02d}_{i:02d}.fastq.gz" for i in range(1, 9)],
    "K562_cells": [f"{i:02d}_{i:02d}.fastq.gz" for i in range(9, 17)],
}

GROUP_COLORS = {"mNeonGreen": "#cff2d8", "mScarlet": "#fad1b2", "NG_BC": "#77DD78", "SC_BC": "#FF6962"}


def plot_stacked_proportions(df: pd.DataFrame, prop_cols: list[str], labels: list[str], title: str, out_base: Path) -> None:
    groups = [c.replace("prop_", "", 1) for c in prop_cols]
    values = df[prop_cols].fillna(0.0).to_numpy(float)
    row_sums = values.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    values = values / row_sums[:, None]

    fig_w = max(6, 0.45 * len(df))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for i, group in enumerate(groups):
        ax.bar(
            x,
            values[:, i],
            bottom=bottom,
            label=group,
            color=GROUP_COLORS.get(group),
            edgecolor="black",
            linewidth=0.4,
        )
        bottom += values[:, i]
    ax.set_ylabel("Proportion of hits")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylim(0, 1.0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Group")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_d06() -> list[str]:
    table = pd.read_csv(TABLE_DIR / "PL47_D06_barcode_hit_table.csv")
    prop_cols = [c for c in table.columns if c.startswith("prop_")]
    outputs = []
    titles = {"T_cells": "T cells", "K562_cells": "K562 cells"}
    for label, fastqs in D06_FASTQ_GROUPS.items():
        subset = table.set_index("fastq").loc[fastqs].reset_index()
        out_base = FIGURE_DIR / "barcode_proportions" / f"PL47_D06_barcode_proportions_{label}"
        plot_stacked_proportions(subset, prop_cols, subset["fastq"].tolist(), titles[label], out_base)
        outputs.extend([relpath(out_base.with_suffix(".pdf")), relpath(out_base.with_suffix(".png"))])
    return outputs


def plot_d05() -> list[str]:
    df = pd.read_csv(TABLE_DIR / "PL47_D05_summary_edit.csv")
    samples = list(dict.fromkeys(df["Sample"].tolist()))
    sorted_values = list(dict.fromkeys(df["Sorted"].tolist()))
    ncols = 4
    nrows = int(np.ceil(len(samples) * len(sorted_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20), squeeze=False)
    axes_flat = axes.ravel()
    i = 0
    for sample in samples:
        for sorted_value in sorted_values:
            ax = axes_flat[i]
            sub = df[(df["Sample"].eq(sample)) & (df["Sorted"].eq(sorted_value))]
            if sub.empty:
                ax.axis("off")
                i += 1
                continue
            vals = [float(sub.iloc[0]["NG_BC"]), float(sub.iloc[0]["SC_BC"])]
            labels = ["NG_BC", "SC_BC"]
            ax.bar(labels, vals, color=[GROUP_COLORS["NG_BC"], GROUP_COLORS["SC_BC"]])
            ax.set_title(f"{sample} {sorted_value}")
            ax.set_ylim(0, 1)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)
            i += 1
    for ax in axes_flat[i:]:
        ax.axis("off")
    fig.tight_layout()
    out_base = FIGURE_DIR / "barcode_proportions" / "PL47_D05_all_plots"
    for ext in ["pdf", "png"]:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [relpath(out_base.with_suffix(".pdf")), relpath(out_base.with_suffix(".png"))]


def convert_flow_pdfs() -> list[str]:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm is required to rasterize the flow cytometry PDFs.")
    outputs = []
    log_lines = []
    flow_dir = TABLE_DIR / "flow_panel_sources"
    for pdf in sorted(flow_dir.glob("*.pdf")):
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", pdf.stem)
        out_png = FIGURE_DIR / "flow_panels" / f"{stem}.png"
        out_pdf = FIGURE_DIR / "flow_panels" / f"{stem}.pdf"
        with tempfile.TemporaryDirectory(prefix="edf4_flow_") as tmp:
            prefix = Path(tmp) / stem
            subprocess.run(
                [pdftoppm, "-singlefile", "-png", "-r", "300", str(pdf), str(prefix)],
                check=True,
                capture_output=True,
                text=True,
            )
            rendered_png = prefix.with_suffix(".png")
            shutil.copyfile(rendered_png, out_png)
        img = mpimg.imread(out_png)
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width / 300, height / 300), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis("off")
        fig.savefig(out_pdf, dpi=300)
        plt.close(fig)
        outputs.extend([relpath(out_pdf), relpath(out_png)])
        log_lines.append(f"Rendered {relpath(pdf)} with pdftoppm at 300 dpi.")
    (SUMMARY_DIR / "flow_pdf_render.log").write_text("\n".join(log_lines) + ("\n" if log_lines else ""))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Extended Data Figure 4 PL47 audit tables and figures.")
    parser.add_argument("--hash-large-files", action="store_true", help="SHA256 hash files larger than 1 GiB.")
    args = parser.parse_args()
    started = utc_now()
    t0 = time.time()
    ensure_dirs()

    raw_manifest = write_manifest(hash_large_files=args.hash_large_files)
    d05_summary = build_d05_tables()
    d06_summary = build_d06_tables()
    figure_outputs = []
    figure_outputs.extend(plot_d05())
    figure_outputs.extend(plot_d06())
    figure_outputs.extend(convert_flow_pdfs())

    pdfs = sorted(relpath(p) for p in FIGURE_DIR.rglob("*.pdf"))
    pngs = sorted(relpath(p) for p in FIGURE_DIR.rglob("*.png"))
    qc = {
        "workflow": "Extended Data Figure 4 PL47 audit workflow",
        "d05": d05_summary,
        "d06": d06_summary,
        "flow_panel_count": len(list((ROOT / "PL47-D04").glob("*.pdf"))),
        "figure_pdfs": pdfs,
        "figure_pngs": pngs,
        "raw_manifest_rows": len(raw_manifest),
        "raw_large_files_hashed": bool(args.hash_large_files),
    }
    write_json(SUMMARY_DIR / "qc_summary.json", qc)

    run_summary = {
        "workflow": "Extended Data Figure 4 PL47 auditable analysis and publication figure generation",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "duration_seconds": round(time.time() - t0, 3),
        "root": ".",
        "script": "run_workflow.py",
        "python": sys.version,
        "package_versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "regex": regex.__version__,
        },
        "outputs": write_output_manifest(),
        "qc_summary": relpath(SUMMARY_DIR / "qc_summary.json"),
    }
    write_json(SUMMARY_DIR / "run_summary.json", run_summary)
    print(f"Workflow complete: {ROOT}")
    print(f"Run summary: {SUMMARY_DIR / 'run_summary.json'}")
    print(f"QC summary:  {SUMMARY_DIR / 'qc_summary.json'}")


if __name__ == "__main__":
    main()
