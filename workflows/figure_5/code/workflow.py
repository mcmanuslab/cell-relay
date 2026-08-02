#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/figure5_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from screen_analysis import run_screen_hit_workflow


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "outputs" / "summaries"
FIGURE_TABLE_DIR = ROOT / "data" / "figure_tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
RPLOT_DIR = FIGURE_DIR / "screen_analysis"

MAGECK_DIR = ROOT / "data" / "upstream" / "mageck"
SCREEN_DIR = ROOT / "data" / "upstream" / "screen_analysis"
VALIDATION_DIR = ROOT / "data" / "upstream" / "validation"
RAW_DIR = ROOT / "data" / "raw"
UPSTREAM_DIR = ROOT / "data" / "upstream"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relpath(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def ensure_dirs() -> None:
    for path in [
        SUMMARY_DIR,
        FIGURE_TABLE_DIR,
        FIGURE_DIR,
        FIGURE_DIR / "barcode_stacks",
        FIGURE_DIR / "decontamination",
        FIGURE_DIR / "validation",
        FIGURE_DIR / "umi_qc",
        RPLOT_DIR,
        FIGURE_DIR / "program_cards",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_manifest(
    paths: list[Path],
    out_path: Path,
    *,
    hash_large_files: bool = False,
    large_file_limit: int = 1024**3,
) -> list[dict]:
    rows = []
    for base in paths:
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
            if hash_large_files or size <= large_file_limit:
                row["sha256"] = sha256_file(path)
            else:
                row["sha256"] = None
                row["sha256_note"] = "not_hashed_by_default_large_file"
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return rows


def strip_mageck_executable(command: str) -> str:
    parts = shlex.split(command)
    for i, part in enumerate(parts):
        if Path(part).name == "mageck":
            parts[i] = "mageck"
            return shlex.join(parts[i:])
    return command


def recover_mageck_commands() -> list[dict]:
    rows = []
    for log_path in sorted(MAGECK_DIR.glob("*.log")):
        command = None
        for line in log_path.read_text(errors="replace").splitlines():
            if "Parameters:" in line:
                command = line.split("Parameters:", 1)[1].strip()
                break
        if command is None:
            continue
        clean_command = strip_mageck_executable(command)
        parts = shlex.split(clean_command)
        input_paths = []
        for flag in ("-k", "-d"):
            if flag in parts and parts.index(flag) + 1 < len(parts):
                input_paths.append(parts[parts.index(flag) + 1])
        rows.append(
            {
                "log_file": relpath(log_path),
                "output_prefix": log_path.stem,
                "command": clean_command,
                "input_paths": ";".join(input_paths),
                "inputs_available": all((MAGECK_DIR / p).exists() for p in input_paths),
            }
        )
    pd.DataFrame(rows).to_csv(SUMMARY_DIR / "mageck_commands.csv", index=False)
    write_json(SUMMARY_DIR / "mageck_commands.json", rows)
    write_mageck_rerun_script(rows)
    return rows


def write_mageck_rerun_script(rows: list[dict]) -> None:
    script = ROOT / "run_mageck_commands.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'WORKDIR="$ROOT/5_mageck_rerun"',
        'mkdir -p "$WORKDIR"',
        'cd "$WORKDIR"',
        'cp "$ROOT/5_mageck/NT466.count.txt" ./',
        'cp "$ROOT/5_mageck/design_matrix.txt" ./',
        'cp "$ROOT/5_mageck/design_matrix_split.txt" ./',
        "",
        "# Recovered MAGeCK commands with available inputs.",
    ]
    for row in rows:
        if row["inputs_available"]:
            lines.append(row["command"])
        else:
            lines.append(f"# Skipped, missing recorded inputs: {row['command']}")
    script.write_text("\n".join(lines) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def load_selected_columns(path: Path, rename_map: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = [col for col in rename_map if col not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")
    return df.loc[:, list(rename_map.keys())].rename(columns=rename_map)


def split_gene_transcript(gene) -> pd.Series:
    if pd.isna(gene):
        return pd.Series([pd.NA, pd.NA], index=["Gene", "Transcript"])
    text = str(gene)
    prefix, sep, suffix = text.partition("_")
    if prefix == "NTC" or sep == "":
        return pd.Series([text, pd.NA], index=["Gene", "Transcript"])
    return pd.Series([prefix, suffix], index=["Gene", "Transcript"])


def open_text_auto(source: Path):
    if str(source).endswith(".gz"):
        return gzip.open(source, "rt", encoding="utf-8")
    return source.open("rt", encoding="utf-8")


def read_ncbi_gene_info(source: Path) -> pd.DataFrame:
    with open_text_auto(source) as fh:
        first_line = fh.readline().strip()
    with open_text_auto(source) as fh:
        if first_line.startswith("#Format:"):
            next(fh)
            header = first_line.replace("#Format:", "", 1).strip()
            columns = header.split("\t") if "\t" in header else header.split()
            gene_info = pd.read_csv(fh, sep="\t", names=columns, dtype=str, keep_default_na=False)
        else:
            gene_info = pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False)
    gene_info = gene_info.replace({"-": pd.NA, "": pd.NA})
    required = {"GeneID", "Symbol"}
    missing = required - set(gene_info.columns)
    if missing:
        raise KeyError(f"NCBI gene_info file is missing required columns: {sorted(missing)}")
    return gene_info


def build_unique_symbol_map(gene_info: pd.DataFrame, symbol_col: str) -> pd.Series:
    if symbol_col not in gene_info.columns:
        return pd.Series(dtype="object")
    lookup = (
        gene_info.loc[:, [symbol_col, "GeneID"]]
        .rename(columns={symbol_col: "match_symbol"})
        .dropna(subset=["match_symbol", "GeneID"])
        .drop_duplicates()
    )
    unique = lookup.groupby("match_symbol")["GeneID"].nunique()
    unique_symbols = unique[unique == 1].index
    return (
        lookup[lookup["match_symbol"].isin(unique_symbols)]
        .drop_duplicates(subset=["match_symbol"])
        .set_index("match_symbol")["GeneID"]
    )


def build_unique_synonym_map(gene_info: pd.DataFrame) -> pd.Series:
    if "Synonyms" not in gene_info.columns:
        return pd.Series(dtype="object")
    syn = gene_info.loc[:, ["GeneID", "Synonyms"]].dropna().copy()
    syn = syn[syn["Synonyms"].ne("-")]
    syn["Synonym"] = syn["Synonyms"].map(lambda x: x.split("|"))
    syn = syn.explode("Synonym")
    syn["Synonym"] = syn["Synonym"].astype("string").str.strip()
    syn = syn[syn["Synonym"].notna() & syn["Synonym"].ne("") & syn["Synonym"].ne("-")]
    syn = syn[["Synonym", "GeneID"]].drop_duplicates()
    unique = syn.groupby("Synonym")["GeneID"].nunique()
    unique_synonyms = unique[unique.eq(1)].index
    return (
        syn[syn["Synonym"].isin(unique_synonyms)]
        .drop_duplicates(subset=["Synonym"])
        .set_index("Synonym")["GeneID"]
    )


def parse_depmap_label(label: str):
    label = str(label).strip()
    match = re.match(r"^(.*?)\s+\((\d+)\)$", label)
    if match:
        return match.group(1), int(match.group(2))
    return label, pd.NA


def get_model_col(columns) -> str:
    preferred = ["ModelID", "depmap_id", "DepMap_ID", "DepMapID"]
    lowered = {str(c).strip().lower(): c for c in columns}
    for name in preferred:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return columns[0]


def extract_k562_depmap_scores(csv_path: Path, model_id: str = "ACH-000551") -> pd.DataFrame:
    non_gene_cols = {
        "",
        "Unnamed: 0",
        "ModelID",
        "ModelConditionID",
        "IsDefaultEntryModel",
        "IsDefaultEntryMC",
        "DepMap_ID",
        "depmap_id",
        "DepMapID",
        "CCLE_Name",
        "CellLineName",
        "StrippedCellLineName",
        "Lineage",
        "OncotreeLineage",
        "PrimaryDisease",
        "Subtype",
    }
    k562_row = None
    for chunk in pd.read_csv(csv_path, chunksize=512, low_memory=False):
        model_col = get_model_col(list(chunk.columns))
        if model_col != "ModelID":
            chunk = chunk.rename(columns={model_col: "ModelID"})
        hit = chunk[chunk["ModelID"].astype(str).str.strip().eq(model_id)]
        if not hit.empty:
            k562_row = hit.iloc[0]
            break
    if k562_row is None:
        raise ValueError(f"K562 / {model_id} not found in {csv_path}")
    records = []
    for col, val in k562_row.items():
        if col in non_gene_cols or str(col).startswith("Unnamed:"):
            continue
        score = pd.to_numeric(val, errors="coerce")
        if pd.isna(score):
            continue
        gene_symbol, entrez_id = parse_depmap_label(col)
        records.append((gene_symbol, entrez_id, float(score), col))
    out = pd.DataFrame(records, columns=["Gene", "DepMap_GeneID", "DepMap", "DepMap_label"])
    out["DepMap_GeneID"] = pd.to_numeric(out["DepMap_GeneID"], errors="coerce").astype("Int64")
    return out


def build_depmap_maps(depmap_scores: pd.DataFrame):
    geneid_map = (
        depmap_scores.dropna(subset=["DepMap_GeneID"])
        .drop_duplicates(subset=["DepMap_GeneID"])
        .set_index("DepMap_GeneID")["DepMap"]
    )
    symbol_counts = depmap_scores.groupby("Gene")["DepMap"].size()
    unique_symbols = symbol_counts[symbol_counts.eq(1)].index
    symbol_map = (
        depmap_scores[depmap_scores["Gene"].isin(unique_symbols)]
        .drop_duplicates(subset=["Gene"])
        .set_index("Gene")["DepMap"]
    )
    return geneid_map, symbol_map


def xlsx_sheet_to_dataframe(path: Path, sheet_name: str, header: int = 0) -> pd.DataFrame:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    def col_idx(cell_ref: str) -> int:
        letters = re.match(r"([A-Z]+)", cell_ref).group(1)
        idx = 0
        for ch in letters:
            idx = idx * 26 + ord(ch) - ord("A") + 1
        return idx - 1

    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                texts = [t.text or "" for t in si.findall(".//main:t", ns)]
                shared.append("".join(texts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall("pkgrel:Relationship", ns)}
        target = None
        for sheet in workbook.findall(".//main:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                target = rel_map[sheet.attrib[f"{{{ns['rel']}}}id"]]
                break
        if target is None:
            raise KeyError(f"Sheet {sheet_name!r} not found in {path}")
        sheet_path = "xl/" + target.lstrip("/")
        root = ET.fromstring(zf.read(sheet_path))

    rows = []
    for row in root.findall(".//main:sheetData/main:row", ns):
        values = []
        for c in row.findall("main:c", ns):
            i = col_idx(c.attrib["r"])
            while len(values) <= i:
                values.append(pd.NA)
            cell_type = c.attrib.get("t")
            value_node = c.find("main:v", ns)
            inline_node = c.find("main:is/main:t", ns)
            if cell_type == "s" and value_node is not None:
                value = shared[int(value_node.text)]
            elif cell_type == "inlineStr" and inline_node is not None:
                value = inline_node.text or ""
            elif value_node is not None:
                value = value_node.text
            else:
                value = pd.NA
            values[i] = value
        rows.append(values)
    if len(rows) <= header:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    rows = [r + [pd.NA] * (width - len(r)) for r in rows]
    columns = [str(c).strip() if pd.notna(c) else f"Unnamed_{i}" for i, c in enumerate(rows[header])]
    return pd.DataFrame(rows[header + 1 :], columns=columns)


def add_surfaceome(combined: pd.DataFrame, surfaceome_file: Path) -> pd.DataFrame:
    ref = xlsx_sheet_to_dataframe(surfaceome_file, "SurfaceomeMasterTable", header=1)
    ref.columns = ref.columns.astype(str).str.strip()
    required = ["GeneID", "UniProt gene", "Surfaceome Label"]
    missing = [c for c in required if c not in ref.columns]
    if missing:
        raise KeyError(f"Missing required columns in {surfaceome_file}: {missing}")
    ref = ref[required].copy()
    ref["GeneID"] = pd.to_numeric(ref["GeneID"], errors="coerce").astype("Int64")
    ref["UniProt gene"] = ref["UniProt gene"].astype("string").str.strip()
    ref["Surfaceome Label"] = ref["Surfaceome Label"].astype("string").str.strip()

    geneid_ref = ref.dropna(subset=["GeneID", "Surfaceome Label"])[["GeneID", "Surfaceome Label"]].drop_duplicates()
    unique_geneids = geneid_ref.groupby("GeneID")["Surfaceome Label"].nunique()
    unique_geneids = unique_geneids[unique_geneids.eq(1)].index
    geneid_map = (
        geneid_ref[geneid_ref["GeneID"].isin(unique_geneids)]
        .drop_duplicates(subset=["GeneID"])
        .set_index("GeneID")["Surfaceome Label"]
    )

    gene_ref = ref.dropna(subset=["UniProt gene", "Surfaceome Label"])[["UniProt gene", "Surfaceome Label"]].drop_duplicates()
    unique_genes = gene_ref.groupby("UniProt gene")["Surfaceome Label"].nunique()
    unique_genes = unique_genes[unique_genes.eq(1)].index
    gene_map = (
        gene_ref[gene_ref["UniProt gene"].isin(unique_genes)]
        .drop_duplicates(subset=["UniProt gene"])
        .set_index("UniProt gene")["Surfaceome Label"]
    )

    if "Surfaceome" in combined.columns:
        combined = combined.drop(columns=["Surfaceome"])
    combined["GeneID"] = pd.to_numeric(combined["GeneID"], errors="coerce").astype("Int64")
    combined["Gene"] = combined["Gene"].astype("string").str.strip()
    combined["Surfaceome"] = combined["GeneID"].map(geneid_map)
    needs_fallback = combined["Surfaceome"].isna() & combined["Gene"].notna()
    combined.loc[needs_fallback, "Surfaceome"] = combined.loc[needs_fallback, "Gene"].map(gene_map)
    return combined[[c for c in combined.columns if c != "Surfaceome"] + ["Surfaceome"]]


def build_combined_table() -> pd.DataFrame:
    mle_rename = {
        "Gene": "Gene",
        "sgRNA": "MLE_sgRNA",
        "treatment|beta": "MLE_beta",
        "treatment|z": "MLE_z",
        "treatment|p-value": "MLE_p-value",
        "treatment|fdr": "MLE_fdr",
    }
    mle_split_rename = {
        "Gene": "Gene",
        "sgRNA": "MLE_S_sgRNA",
        "B-K|beta": "MLE_B_beta",
        "B-K|z": "MLE_B_z",
        "B-K|p-value": "MLE_B_p-value",
        "B-K|fdr": "MLE_B_fdr",
        "C-K|beta": "MLE_C_beta",
        "C-K|z": "MLE_C_z",
        "C-K|p-value": "MLE_C_p-value",
        "C-K|fdr": "MLE_C_fdr",
    }
    b_rename = {
        "id": "Gene",
        "neg|lfc": "B_lfc",
        "neg|p-value": "B_neg_p-value",
        "neg|fdr": "B_neg_fdr",
        "neg|rank": "B_neg_rank",
        "neg|goodsgrna": "B_neg_goodsgrna",
        "pos|p-value": "B_pos_p-value",
        "pos|fdr": "B_pos_fdr",
        "pos|rank": "B_pos_rank",
        "pos|goodsgrna": "B_pos_goodsgrna",
    }
    c_rename = {k: v.replace("B_", "C_") for k, v in b_rename.items()}
    c_rename["id"] = "Gene"

    mle = load_selected_columns(MAGECK_DIR / "NT466_mle.gene_summary.txt", mle_rename)
    mle_split = load_selected_columns(MAGECK_DIR / "NT466_mle_split.gene_summary.txt", mle_split_rename)
    b = load_selected_columns(MAGECK_DIR / "B.gene_summary.txt", b_rename)
    c = load_selected_columns(MAGECK_DIR / "C.gene_summary.txt", c_rename)
    for label, df in [("B", b), ("C", c)]:
        dupes = df.loc[df["Gene"].duplicated(), "Gene"].dropna().unique().tolist()
        if dupes:
            raise ValueError(f"{label} summary has duplicated Gene values: {dupes[:10]}")

    combined = (
        mle.merge(mle_split, on="Gene", how="left", validate="many_to_one")
        .merge(b, on="Gene", how="left", validate="many_to_one")
        .merge(c, on="Gene", how="left", validate="many_to_one")
    )
    gene_tx = combined["Gene"].apply(split_gene_transcript)
    combined["Gene"] = gene_tx["Gene"].astype("string")
    combined.insert(1, "Transcript", gene_tx["Transcript"].astype("string"))

    gene_info = read_ncbi_gene_info(SCREEN_DIR / "ref" / "Homo_sapiens.gene_info.gz")
    symbol_to_geneid = build_unique_symbol_map(gene_info, "Symbol_from_nomenclature_authority").combine_first(
        build_unique_symbol_map(gene_info, "Symbol")
    )
    combined.insert(1, "GeneID", combined["Gene"].map(symbol_to_geneid))
    is_ntc = combined["Gene"].eq("NTC") | combined["Gene"].str.startswith("NTC_")
    combined.loc[is_ntc, "GeneID"] = pd.NA
    combined["GeneID"] = pd.to_numeric(combined["GeneID"], errors="coerce").astype("Int64")

    synonym_map = build_unique_synonym_map(gene_info)
    needs_syn = combined["GeneID"].isna() & ~is_ntc & combined["Gene"].notna()
    synonym_hits = pd.to_numeric(combined.loc[needs_syn, "Gene"].map(synonym_map), errors="coerce").astype("Int64")
    combined.loc[synonym_hits.index[synonym_hits.notna()], "GeneID"] = synonym_hits[synonym_hits.notna()]

    final_cols = [
        "Gene",
        "GeneID",
        "Transcript",
        "MLE_sgRNA",
        "MLE_beta",
        "MLE_z",
        "MLE_p-value",
        "MLE_fdr",
        "MLE_S_sgRNA",
        "MLE_B_beta",
        "MLE_B_z",
        "MLE_B_p-value",
        "MLE_B_fdr",
        "MLE_C_beta",
        "MLE_C_z",
        "MLE_C_p-value",
        "MLE_C_fdr",
        "B_lfc",
        "B_neg_p-value",
        "B_neg_fdr",
        "B_neg_rank",
        "B_neg_goodsgrna",
        "B_pos_p-value",
        "B_pos_fdr",
        "B_pos_rank",
        "B_pos_goodsgrna",
        "C_lfc",
        "C_neg_p-value",
        "C_neg_fdr",
        "C_neg_rank",
        "C_neg_goodsgrna",
        "C_pos_p-value",
        "C_pos_fdr",
        "C_pos_rank",
        "C_pos_goodsgrna",
    ]
    combined = combined.loc[:, final_cols]

    depmap_scores = extract_k562_depmap_scores(SCREEN_DIR / "depmap_cache" / "CRISPRGeneEffect.csv")
    geneid_map, symbol_map = build_depmap_maps(depmap_scores)
    combined.insert(combined.columns.get_loc("GeneID") + 1, "DepMap", pd.NA)
    combined["GeneID"] = pd.to_numeric(combined["GeneID"], errors="coerce").astype("Int64")
    combined["DepMap"] = combined["GeneID"].map(geneid_map)
    is_ntc = combined["Gene"].fillna("").eq("NTC") | combined["Gene"].fillna("").str.startswith("NTC_")
    symbol_candidates = combined["DepMap"].isna() & ~is_ntc & combined["Gene"].notna()
    combined.loc[symbol_candidates, "DepMap"] = combined.loc[symbol_candidates, "Gene"].map(symbol_map)
    combined["DepMap"] = pd.to_numeric(combined["DepMap"], errors="coerce")
    combined = combined[[c for c in combined.columns if c != "DepMap"] + ["DepMap"]]
    combined = add_surfaceome(combined, SCREEN_DIR / "ref" / "table_S3_surfaceome.xlsx")

    out = SCREEN_DIR / "NT466_combined.csv"
    combined.to_csv(out, index=False)
    combined.to_csv(FIGURE_TABLE_DIR / "NT466_combined.csv", index=False)
    return combined


def write_hit_tables(combined: pd.DataFrame) -> dict:
    hits_dir = SCREEN_DIR / "hits"
    neg = combined[(combined["B_neg_fdr"] < 0.05) & (combined["C_neg_fdr"] < 0.05)].copy()
    pos = combined[(combined["B_pos_fdr"] < 0.05) & (combined["C_pos_fdr"] < 0.05)].copy()
    neg_depmap = neg[neg["DepMap"] > -0.25].copy()
    pos_depmap = pos[pos["DepMap"] > -0.25].copy()
    tables = {
        "hits_neg": neg,
        "hits_pos": pos,
        "hits_neg_depmap": neg_depmap,
        "hits_pos_depmap": pos_depmap,
    }
    for name, df in tables.items():
        df.to_csv(hits_dir / f"{name}.tsv", sep="\t", index=False)
        df.to_csv(FIGURE_TABLE_DIR / f"{name}.tsv", sep="\t", index=False)
    return {name: int(len(df)) for name, df in tables.items()}


def build_dotplot_table(combined: pd.DataFrame) -> pd.DataFrame:
    df = combined[np.isfinite(combined["B_lfc"]) & np.isfinite(combined["C_lfc"])].copy()
    df["Gene_clean"] = df["Gene"].astype(str).str.replace(r"_.*$", "", regex=True)
    df["Category"] = np.select(
        [
            (df["B_neg_fdr"] < 0.05) & (df["C_neg_fdr"] < 0.05),
            (df["B_pos_fdr"] < 0.05) & (df["C_pos_fdr"] < 0.05),
        ],
        ["neg_sig", "pos_sig"],
        default="other",
    )
    df.to_csv(FIGURE_TABLE_DIR / "B_vs_C_dotplot_table.csv", index=False)
    return df


def plot_b_vs_c_dotplot(table: pd.DataFrame) -> None:
    table_path = FIGURE_TABLE_DIR / "B_vs_C_dotplot_table.csv"
    if not table_path.exists():
        table.to_csv(table_path, index=False)
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required to recreate NT466_B_vs_C_dotplot with the original R plotting workflow.")
    script = ROOT / "code" / "NT466_4.3_Rplots.r"
    command = [
        rscript,
        str(script),
        str(table_path),
        str(RPLOT_DIR / "NT466_B_vs_C_dotplot.pdf"),
        str(RPLOT_DIR / "NT466_B_vs_C_dotplot.png"),
        str(RPLOT_DIR / "NT466_B_vs_C_dotplot.svg"),
        str(SUMMARY_DIR / "B_vs_C_dotplot_labels_generated.csv"),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (SUMMARY_DIR / "NT466_4.3_Rplots.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"R dotplot generation failed; see {SUMMARY_DIR / 'NT466_4.3_Rplots.log'}")


def parse_gene_base(gene: str) -> str:
    if pd.isna(gene):
        return ""
    return str(gene).split("_")[0]


def load_sgrna_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"Gene", "LFC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.copy()
    df["Gene"] = df["Gene"].astype(str)
    df["gene_base"] = df["Gene"].map(parse_gene_base)
    df["LFC"] = pd.to_numeric(df["LFC"], errors="coerce")
    return df.dropna(subset=["LFC", "Gene"])


def make_kde(values, x_grid):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        y = np.zeros_like(x_grid, dtype=float)
        if len(values):
            y[np.argmin(np.abs(x_grid - values[0]))] = 1.0
        return y
    return gaussian_kde(values)(x_grid)


def plot_barcode_stack(path: Path | None = None, *, table: pd.DataFrame | None = None, label: str | None = None) -> None:
    target_genes = ["B2M", "HLA-A", "TAP1", "TAP2", "TAPBP", "PDIA3", "CALR", "PSMB8", "PSMB9"]
    gene_colors = {
        "NTC": "#bdbdbd",
        "B2M": "#bdbdbd",
        "HLA-A": "#bdbdbd",
        "TAP1": "#a68ff8",
        "TAP2": "#a68ff8",
        "TAPBP": "#74b4f9",
        "PDIA3": "#78e1f6",
        "CALR": "#78e1f6",
        "PSMB8": "#ed8590",
        "PSMB9": "#ed8590",
    }
    if table is None:
        if path is None:
            raise ValueError("Provide either path or table.")
        df = load_sgrna_summary(path)
        label = label or path.stem
    else:
        df = table.copy()
        if "gene_base" not in df:
            df["gene_base"] = df["Gene"].map(parse_gene_base)
        if label is None:
            raise ValueError("A label is required when plotting from a table.")
    xlim = (-3, 3)
    x_grid = np.linspace(xlim[0], xlim[1], 1000)
    nrows = 2 + len(target_genes)
    height_ratios = [1.2, 0.55] + [0.55] * len(target_genes)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(5, sum(height_ratios) + 0.6),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.12},
    )
    y = make_kde(df["LFC"].values, x_grid)
    axes[0].fill_between(x_grid, y, color="#9e9e9e")
    axes[0].plot(x_grid, y, color="black", linewidth=1.4)
    axes[0].set_yticks([])

    ntc = df.loc[df["Gene"].str.startswith("NTC_"), "LFC"].values
    axes[1].vlines(ntc, 0, 1, color=gene_colors["NTC"], linewidth=1.1, alpha=0.45)
    style_bar_axis(axes[1], "NO-TARGET")
    for i, gene in enumerate(target_genes, start=2):
        values = df.loc[df["gene_base"] == gene, "LFC"].values
        axes[i].vlines(values, 0, 1, color=gene_colors.get(gene, "#333333"), linewidth=3.1)
        style_bar_axis(axes[i], gene)
    for ax in axes:
        ax.set_xlim(*xlim)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
    axes[-1].set_xlabel("LFC", fontsize=13)
    fig.suptitle(label, y=0.995, fontsize=14)
    out_base = FIGURE_DIR / "barcode_stacks" / f"{label}_lfc_barcode_stack"
    for ext in ["pdf", "png"]:
        fig.savefig(f"{out_base}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def style_bar_axis(ax, label: str) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=34, fontsize=12)


def load_barcode_totals(path: Path, barcode_col: str = "ID", count_col: str = "count") -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[barcode_col, count_col])
    df = df.rename(columns={barcode_col: "barcode", count_col: "count"})
    df["barcode"] = df["barcode"].astype(str)
    out = df.groupby("barcode", as_index=False)["count"].sum()
    out["count"] = out["count"].astype(float)
    return out.sort_values("barcode").reset_index(drop=True)


def add_lfc(merged: pd.DataFrame, ref_lib: float, query_lib: float, query_col: str, pseudocount: float = 0.5):
    n = merged.shape[0]
    ref_cpm = 1e6 * (merged["reference_count"] + pseudocount) / (ref_lib + pseudocount * n)
    qry_cpm = 1e6 * (merged[query_col] + pseudocount) / (query_lib + pseudocount * n)
    return np.log2(qry_cpm / ref_cpm)


def build_decontam_lfc_tables() -> pd.DataFrame:
    ref = load_barcode_totals(UPSTREAM_DIR / "id_umi_counts_collapsed" / "NT466_K__collapsed_hd3_ratio10.csv")
    ref_lib = ref["count"].sum()
    rows = []
    specs = [
        ("NT466_B", "NT466_B__collapsed_hd3_ratio10.csv"),
        ("NT466_C", "NT466_C__collapsed_hd3_ratio10.csv"),
        ("NT466_D", "NT466_D__collapsed_hd3_ratio10.csv"),
    ]
    for label, filename in specs:
        before = load_barcode_totals(UPSTREAM_DIR / "id_umi_counts_collapsed" / filename)
        after = load_barcode_totals(UPSTREAM_DIR / "decontamination" / filename.replace(".csv", "") / "recipient_pairs_cleaned.csv")
        base = pd.DataFrame({"barcode": sorted(set(ref["barcode"]))})
        merged = (
            base.merge(ref.rename(columns={"count": "reference_count"}), on="barcode", how="left")
            .merge(before.rename(columns={"count": "before_count"}), on="barcode", how="left")
            .merge(after.rename(columns={"count": "after_count"}), on="barcode", how="left")
            .fillna(0.0)
        )
        merged["comparison"] = label
        merged["before_lfc_vs_reference"] = add_lfc(merged, ref_lib, before["count"].sum(), "before_count")
        merged["after_lfc_vs_reference"] = add_lfc(merged, ref_lib, after["count"].sum(), "after_count")
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(FIGURE_TABLE_DIR / "decontamination_lfc_table.csv", index=False)
    return out


def plot_decontam_density(table: pd.DataFrame) -> None:
    vals = np.concatenate([table["before_lfc_vs_reference"].to_numpy(), table["after_lfc_vs_reference"].to_numpy()])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.quantile(vals, [0.001, 0.999])
    pad = 0.05 * (hi - lo)
    xlim = (lo - pad, hi + pad)
    x_grid = np.linspace(xlim[0], xlim[1], 512)
    summary = []
    for label, sub in table.groupby("comparison", sort=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for col, color, name in [
            ("before_lfc_vs_reference", "#7a716b", "before"),
            ("after_lfc_vs_reference", "#5d8af7", "after"),
        ]:
            values = sub[col].to_numpy(float)
            values = values[np.isfinite(values)]
            y = make_kde(values, x_grid)
            ax.plot(x_grid, y, color=color, linewidth=2, label=name)
            ax.fill_between(x_grid, y, color=color, alpha=0.18)
            summary.append(
                {
                    "comparison": label,
                    "sample": name,
                    "n_barcodes": int(len(values)),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "p99": float(np.quantile(values, 0.99)),
                }
            )
        ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xlim(*xlim)
        ax.set_xlabel("log2 fold change vs K reference")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURE_DIR / "decontamination" / f"{label}_before_after_lfc_density.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(summary).to_csv(SUMMARY_DIR / "decontamination_lfc_summary_generated.csv", index=False)


def plot_validation_heatmaps() -> dict:
    df = pd.read_csv(VALIDATION_DIR / "validation_heatmap_data.csv")
    df["Gene"] = df["Gene"].astype(str)
    df = df.drop_duplicates("Gene", keep="first")
    plot_df = pd.DataFrame(
        {
            "Gene": df["Gene"],
            "K562|Trogocytosis": df[["TR_LFC_K562_1", "TR_LFC_K562_2"]].mean(axis=1, skipna=True),
            "K562|IFN-gamma": df[["IFN_LFC_K562_1", "IFN_LFC_K562_2"]].mean(axis=1, skipna=True),
            "K562|HLA-A2": df["HLA_LFC_K562"],
            "K562|CD81": df["CD81_LFC_K562"],
            "293T|Trogocytosis": df[["TR_LFC_293T_1", "TR_LFC_293T_2"]].mean(axis=1, skipna=True),
            "293T|HLA-A2": df["HLA_LFC_293T"],
            "293T|CD81": df["CD81_LFC_293T"],
        }
    ).set_index("Gene")
    k562 = plot_df[["K562|Trogocytosis", "K562|IFN-gamma", "K562|HLA-A2", "K562|CD81"]].copy()
    k562.columns = ["Trogocytosis", r"IFN-$\gamma$", "HLA-A2", "CD81"]
    t293 = plot_df[["293T|Trogocytosis", "293T|HLA-A2", "293T|CD81"]].copy()
    t293.columns = ["Trogocytosis", "HLA-A2", "CD81"]
    k562.to_csv(FIGURE_TABLE_DIR / "validation_heatmap_K562_table.csv")
    t293.to_csv(FIGURE_TABLE_DIR / "validation_heatmap_293T_table.csv")
    cmap = LinearSegmentedColormap.from_list("validation_diverging", ["#5d8af7", "#f7f7f7", "#ed8590"], N=256)

    def draw(data: pd.DataFrame, title: str, name: str):
        cell_size = 0.42
        fig_w = max(4.5, data.shape[1] * cell_size + 2.8)
        fig_h = max(6.0, data.shape[0] * cell_size + 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        im = ax.imshow(data.to_numpy(), cmap=cmap, vmin=-2, vmax=2, interpolation="none", aspect="equal")
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
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("LFC", fontsize=10)
        fig.subplots_adjust(left=0.22, right=0.88, top=0.93, bottom=0.06)
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURE_DIR / "validation" / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    draw(k562, "K562", "validation_heatmap_K562")
    draw(t293, "293T", "validation_heatmap_293T")
    return {"K562_rows": int(k562.shape[0]), "293T_rows": int(t293.shape[0])}


def plot_umi_qc(umi_dir: Path | None = None) -> dict:
    umi_dir = umi_dir or FIGURE_TABLE_DIR
    files = sorted(umi_dir.glob("*__perID_UMI_counts.csv"))
    if not files:
        return {"files": 0}
    groups = []
    labels = []
    rows = []
    for path in files:
        df = pd.read_csv(path)
        values = pd.to_numeric(df["UMI_count"], errors="coerce").dropna().to_numpy()
        groups.append(values)
        labels.append(path.name.replace("__perID_UMI_counts.csv", ""))
        rows.append({"file": path.name, "n": int(len(values)), "median": float(np.median(values)), "mean": float(np.mean(values)), "p95": float(np.quantile(values, 0.95))})
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(values, bins="auto", color="#84c7ff", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("UMI_count per ID")
        ax.set_ylabel("Frequency")
        ax.set_title(labels[-1])
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURE_DIR / "umi_qc" / f"{labels[-1]}_UMI_count_hist.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(groups) + 2), 5))
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylabel("UMI_count")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURE_DIR / "umi_qc" / f"UMI_count_per_ID_all_files.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(SUMMARY_DIR / "UMI_count_per_ID_summary_generated.csv", index=False)
    return {"files": int(len(files))}


def output_manifest() -> list[dict]:
    targets = [FIGURE_TABLE_DIR, FIGURE_DIR, SUMMARY_DIR]
    rows = []
    for base in targets:
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path.name in {"run_summary.json", ".DS_Store"}:
                continue
            rows.append({"relative_path": relpath(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def publication_figure_paths(extension: str) -> list[str]:
    roots = [FIGURE_DIR]
    paths = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob(f"*.{extension}"))
    return sorted(relpath(path) for path in paths)


def compare_generated_to_reference() -> dict:
    checks = {}
    for reference_rel, generated_rel in [
        ("6_screen_analysis/NT466_combined.csv", "data/figure_tables/NT466_combined.csv"),
        ("6_screen_analysis/hits/hits_neg.tsv", "data/figure_tables/hits_neg.tsv"),
        ("6_screen_analysis/hits/hits_pos.tsv", "data/figure_tables/hits_pos.tsv"),
        ("6_screen_analysis/hits/hits_neg_depmap.tsv", "data/figure_tables/hits_neg_depmap.tsv"),
        ("6_screen_analysis/hits/hits_pos_depmap.tsv", "data/figure_tables/hits_pos_depmap.tsv"),
    ]:
        reference = ROOT / reference_rel
        generated = ROOT / generated_rel
        if reference.exists() and generated.exists():
            reference_sep = "\t" if reference_rel.endswith(".tsv") else ","
            generated_sep = "\t" if generated_rel.endswith(".tsv") else ","
            a = pd.read_csv(reference, sep=reference_sep)
            b = pd.read_csv(generated, sep=generated_sep)
            checks[reference_rel] = {
                "reference_shape": list(a.shape),
                "generated_shape": list(b.shape),
                "columns_match": list(a.columns) == list(b.columns),
            }
            common = [c for c in a.columns if c in b.columns]
            if common:
                checks[reference_rel]["exact_csv_match"] = a[common].fillna("").astype(str).equals(b[common].fillna("").astype(str))
                if "Gene" in a.columns and "Gene" in b.columns:
                    checks[reference_rel]["gene_set_match"] = sorted(a["Gene"].astype(str)) == sorted(b["Gene"].astype(str))
                    checks[reference_rel]["gene_order_match"] = a["Gene"].astype(str).tolist() == b["Gene"].astype(str).tolist()
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Figure 5 audit tables and publication figures.")
    parser.add_argument("--hash-large-files", action="store_true", help="SHA256 hash FASTQs and other files larger than 1 GiB.")
    parser.add_argument("--skip-decontam-density", action="store_true", help="Skip barcode-level before/after density plots.")
    parser.add_argument("--refresh-mygene", action="store_true", help="Refresh the frozen MyGene annotation snapshot used for hit programs.")
    parser.add_argument("--screen-analysis-only", action="store_true", help="Regenerate only outputs affected by the updated 6_screen_analysis code.")
    args = parser.parse_args()

    started = time.time()
    ensure_dirs()

    if args.screen_analysis_only:
        combined = pd.read_csv(SCREEN_DIR / "NT466_combined.csv")
        dotplot_table = build_dotplot_table(combined)
        plot_b_vs_c_dotplot(dotplot_table)
        screen_summary = run_screen_hit_workflow(refresh_mygene=args.refresh_mygene)
        qc_path = SUMMARY_DIR / "qc_summary.json"
        qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
        qc.pop("string_network", None)
        qc.pop("module_cards", None)
        qc.update(
            {
                "workflow": "Figure 5 NT466 audit workflow",
                "combined_rows": int(len(combined)),
                "dotplot_rows": int(len(dotplot_table)),
                "screen_analysis": screen_summary,
                "figure_pdfs": publication_figure_paths("pdf"),
                "figure_pngs": publication_figure_paths("png"),
            }
        )
        write_json(qc_path, qc)
        outputs = output_manifest()
        run_summary = {
            "workflow": "Figure 5 NT466 auditable analysis and publication figure generation",
            "mode": "screen-analysis-only",
            "root": ".",
            "script": Path(__file__).name,
            "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "finished_at_utc": utc_now(),
            "duration_seconds": round(time.time() - started, 3),
            "python": sys.version,
            "package_versions": {
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "outputs": outputs,
            "qc_summary": "summaries/qc_summary.json",
        }
        write_json(SUMMARY_DIR / "run_summary.json", run_summary)
        print(f"Screen-analysis update complete: {ROOT}")
        print(f"Run summary: {SUMMARY_DIR / 'run_summary.json'}")
        print(f"QC summary:  {qc_path}")
        return

    raw_manifest = write_manifest(
        [
            RAW_DIR,
            SCREEN_DIR / "ref",
            SCREEN_DIR / "depmap_cache" / "CRISPRGeneEffect.csv",
            VALIDATION_DIR,
        ],
        SUMMARY_DIR / "raw_data_manifest.csv",
        hash_large_files=args.hash_large_files,
    )
    mageck_commands = recover_mageck_commands()
    combined = build_combined_table()
    hit_counts = write_hit_tables(combined)
    dotplot_table = build_dotplot_table(combined)
    plot_b_vs_c_dotplot(dotplot_table)
    for path in [MAGECK_DIR / "B.sgrna_summary.txt", MAGECK_DIR / "C.sgrna_summary.txt"]:
        plot_barcode_stack(path)
    decontam_summary = {"skipped": bool(args.skip_decontam_density)}
    if not args.skip_decontam_density:
        decontam_table = build_decontam_lfc_tables()
        plot_decontam_density(decontam_table)
        decontam_summary = {"rows": int(len(decontam_table)), "comparisons": int(decontam_table["comparison"].nunique())}
    screen_summary = run_screen_hit_workflow(refresh_mygene=args.refresh_mygene)
    validation_summary = plot_validation_heatmaps()
    umi_summary = plot_umi_qc(UPSTREAM_DIR / "umi_analyses")

    qc = {
        "workflow": "Figure 5 NT466 audit workflow",
        "raw_manifest_rows": len(raw_manifest),
        "raw_large_files_hashed": bool(args.hash_large_files),
        "mageck_commands_recovered": len(mageck_commands),
        "combined_rows": int(len(combined)),
        "hit_counts": hit_counts,
        "dotplot_rows": int(len(dotplot_table)),
        "decontamination": decontam_summary,
        "screen_analysis": screen_summary,
        "validation_heatmaps": validation_summary,
        "umi_qc": umi_summary,
        "reference_comparisons": compare_generated_to_reference(),
        "figure_pdfs": publication_figure_paths("pdf"),
        "figure_pngs": publication_figure_paths("png"),
    }
    write_json(SUMMARY_DIR / "qc_summary.json", qc)
    outputs = output_manifest()
    run_summary = {
        "workflow": "Figure 5 NT466 auditable analysis and publication figure generation",
        "mode": "full-downstream",
        "root": ".",
        "script": Path(__file__).name,
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at_utc": utc_now(),
        "duration_seconds": round(time.time() - started, 3),
        "python": sys.version,
        "package_versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": outputs,
        "qc_summary": "summaries/qc_summary.json",
    }
    write_json(SUMMARY_DIR / "run_summary.json", run_summary)
    print(f"Workflow complete: {ROOT}")
    print(f"Run summary: {SUMMARY_DIR / 'run_summary.json'}")
    print(f"QC summary:  {SUMMARY_DIR / 'qc_summary.json'}")


if __name__ == "__main__":
    main()
