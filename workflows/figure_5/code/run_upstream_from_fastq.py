#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import betabinom, binom


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
UPSTREAM_DIR = ROOT / "data" / "upstream"
GENE_SEP = "|"


@dataclass
class DecontamParams:
    pseudocount: float = 0.5
    min_recipient_count: int = 100
    min_enrichment: float = 5.0
    min_excess: float = 100.0
    fdr: float = 1e-2
    min_barcode_total: int = 10
    min_donor_ids: int = 10
    clean_mode: str = "zero"
    clean_q: float = 0.95
    max_iter: int = 10
    rho_cap: float = 0.20
    hard_count_cutoff: Optional[int] = 2000
    verbose: bool = True


def load_bc_table(path: Path) -> pd.DataFrame:
    records = []
    with path.open("rt") as fh:
        for line_num, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                parts = line.split("\t", 2)
            elif line.count(",") >= 2:
                parts = line.split(",", 2)
            else:
                parts = line.split(None, 2)
            if len(parts) != 3:
                raise ValueError(f"{path} line {line_num}: expected 3 columns, got {len(parts)}")
            id_, grna, gene = (p.strip() for p in parts)
            records.append((id_, grna.upper(), gene))
    bc = pd.DataFrame(records, columns=["ID", "gRNA", "gene"])
    if bc.empty:
        raise ValueError(f"No records found in {path}")
    if bc["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in BC.txt")
    if bc["gRNA"].duplicated().any():
        raise ValueError("Duplicate gRNA sequences found in BC.txt")
    lengths = bc["gRNA"].str.len().unique()
    if len(lengths) != 1:
        raise ValueError(f"gRNA sequences have multiple lengths: {sorted(lengths)}")
    return bc


def make_matcher(bc: pd.DataFrame, max_mismatches: int = 2):
    grna_len = int(bc["gRNA"].str.len().iat[0])
    base = grna_len // (max_mismatches + 1)
    rem = grna_len % (max_mismatches + 1)
    bounds = []
    start = 0
    for i in range(max_mismatches + 1):
        end = start + base + (1 if i < rem else 0)
        bounds.append((start, end))
        start = end

    ref_seqs = bc["gRNA"].tolist()
    ref_ids = bc["ID"].tolist()
    ref_genes = bc["gene"].tolist()
    seed_index = [defaultdict(list) for _ in bounds]
    for idx, seq in enumerate(ref_seqs):
        for chunk_i, (s, e) in enumerate(bounds):
            seed_index[chunk_i][seq[s:e]].append(idx)

    @lru_cache(maxsize=500_000)
    def match_grna(query: str):
        query = query.upper()
        if len(query) != grna_len or set(query) - set("ACGT"):
            return None
        candidate_idx = set()
        for chunk_i, (s, e) in enumerate(bounds):
            candidate_idx.update(seed_index[chunk_i].get(query[s:e], []))
        best_dist = max_mismatches + 1
        best_hits = []
        for idx in candidate_idx:
            dist = sum(a != b for a, b in zip(query, ref_seqs[idx]))
            if dist > max_mismatches:
                continue
            if dist < best_dist:
                best_dist = dist
                best_hits = [idx]
            elif dist == best_dist:
                best_hits.append(idx)
        if len(best_hits) != 1:
            return None
        i = best_hits[0]
        return ref_ids[i], ref_genes[i]

    return match_grna, grna_len


def iter_fastq_sequences(path: Path):
    with gzip.open(path, "rt") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().strip().upper()
            fh.readline()
            qual = fh.readline()
            if not qual:
                raise ValueError(f"Incomplete FASTQ record in {path}")
            yield seq


def strip_fastq_suffix(name: str) -> str:
    for suffix in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def build_id_umi_counts() -> None:
    bc = load_bc_table(RAW_DIR / "reference" / "BC.txt")
    match_grna, grna_len = make_matcher(bc)
    umi_start, umi_len, grna_start = 0, 16, 20
    grna_end = grna_start + grna_len
    out_dir = UPSTREAM_DIR / "id_umi_counts"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for input_dir in [RAW_DIR / "fastq" / "NT466-D04", RAW_DIR / "fastq" / "NT466-D05"]:
        for path in sorted(input_dir.glob("*.fastq.gz")):
            groups[path.name].append(path)
    if not groups:
        raise FileNotFoundError("No FASTQ files found under data/raw/fastq/NT466-D04 or NT466-D05")

    summary_rows = []
    for filename, paths in sorted(groups.items()):
        pair_counts = Counter()
        stats = Counter()
        for path in paths:
            print(f"reading {path}")
            for seq in iter_fastq_sequences(path):
                stats["total_reads"] += 1
                if len(seq) < max(umi_start + umi_len, grna_end):
                    stats["too_short"] += 1
                    continue
                umi = seq[umi_start : umi_start + umi_len]
                grna = seq[grna_start:grna_end]
                hit = match_grna(grna)
                if hit is None:
                    stats["unmatched_or_ambiguous"] += 1
                    continue
                id_, gene = hit
                pair_counts[(id_, gene, umi)] += 1
                stats["matched_reads"] += 1
        rows = [{"ID": i, "gene": g, "UMI": u, "count": c} for (i, g, u), c in pair_counts.items()]
        out_df = pd.DataFrame(rows, columns=["ID", "gene", "UMI", "count"])
        if not out_df.empty:
            out_df = out_df.sort_values(["ID", "UMI"]).reset_index(drop=True)
        out_path = out_dir / f"{strip_fastq_suffix(filename)}.id_umi_counts.csv"
        out_df.to_csv(out_path, index=False)
        summary_rows.append(
            {
                "input_filename": filename,
                "n_input_files_merged": len(paths),
                "reads_total": stats["total_reads"],
                "reads_matched": stats["matched_reads"],
                "reads_unmatched_or_ambiguous": stats["unmatched_or_ambiguous"],
                "reads_too_short": stats["too_short"],
                "unique_ID_UMI_pairs": len(out_df),
                "output_csv": str(out_path.relative_to(ROOT)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "processing_summary.csv", index=False)


def merge_count_tables() -> None:
    out_dir = UPSTREAM_DIR / "merged_id_umi_counts"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_inputs = {
        "NT466_B": ["NT466_B1_B.id_umi_counts.csv", "NT466_B2_B.id_umi_counts.csv"],
        "NT466_C": ["NT466_C1_B.id_umi_counts.csv", "NT466_C2_B.id_umi_counts.csv"],
        "NT466_D": ["NT466_D1_B.id_umi_counts.csv", "NT466_D2_B.id_umi_counts.csv"],
        "NT466_K": ["NT466_K1_B.id_umi_counts.csv", "NT466_K2_B.id_umi_counts.csv"],
    }
    for sample, names in sample_inputs.items():
        dfs = []
        for name in names:
            df = pd.read_csv(UPSTREAM_DIR / "id_umi_counts" / name)
            df["ID"] = df["ID"].astype(str)
            df["UMI"] = df["UMI"].astype(str)
            dfs.append(df.groupby(["ID", "UMI", "gene"], as_index=False)["count"].sum())
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.groupby(["ID", "UMI"], as_index=False).agg({"count": "sum", "gene": "first"})
        merged = merged.sort_values(["ID", "count"], ascending=[True, False])
        merged[["ID", "gene", "UMI", "count"]].to_csv(out_dir / f"{sample}.csv", index=False)


def collapse_umis_hd_ratio(df: pd.DataFrame, max_hd: int = 3, min_ratio: int = 10):
    out = df.copy()
    out["count"] = pd.to_numeric(out["count"], errors="raise")
    out["UMI"] = out["UMI"].astype(str).str.upper()
    out = out.reset_index(drop=True)
    collapsed_into = np.empty(len(out), dtype=object)
    collapsed_into[:] = None
    collapsed_rows = []
    for gkey, idx in out.groupby(["ID"], sort=False).groups.items():
        g = out.loc[idx].sort_values("count", ascending=False, kind="mergesort")
        umis = g["UMI"].tolist()
        counts = g["count"].to_numpy(dtype=np.int64)
        pos = g.index.to_numpy()
        n = len(umis)
        if n < 2:
            if n == 1:
                collapsed_rows.append(g.iloc[0].copy())
            continue
        lengths = {len(u) for u in umis}
        if len(lengths) != 1:
            raise ValueError(f"group {gkey}: UMIs have different lengths {sorted(lengths)}")
        length = lengths.pop()
        umi_bytes = np.frombuffer("".join(umis).encode("ascii"), dtype=np.uint8).reshape(n, length)
        parent = np.arange(n, dtype=np.int64)
        for i in range(1, n):
            threshold = min_ratio * counts[i]
            cand_end = np.searchsorted(-counts[:i], -threshold, side="right")
            if cand_end <= 0:
                continue
            d = np.count_nonzero(umi_bytes[:cand_end] != umi_bytes[i], axis=1)
            ok = d <= max_hd
            if not ok.any():
                continue
            min_d = int(d[ok].min())
            parent[i] = int(np.flatnonzero(ok & (d == min_d))[0])

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return int(i)

        roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
        for i in range(n):
            r = int(roots[i])
            if r != i:
                collapsed_into[pos[i]] = umis[r]
        root_sums = np.bincount(roots, weights=counts, minlength=n).astype(np.int64)
        for r in np.flatnonzero(roots == np.arange(n, dtype=np.int64)):
            row = g.iloc[int(r)].copy()
            row["count"] = int(root_sums[int(r)])
            collapsed_rows.append(row)
    collapsed = pd.DataFrame(collapsed_rows)
    if not collapsed.empty:
        collapsed = collapsed[out.columns].reset_index(drop=True)
    annotated = out.copy()
    annotated["collapsed_into"] = collapsed_into
    return collapsed, annotated


def collapse_merged_counts() -> None:
    in_dir = UPSTREAM_DIR / "merged_id_umi_counts"
    out_dir = UPSTREAM_DIR / "id_umi_counts_collapsed"
    umi_dir = UPSTREAM_DIR / "umi_analyses"
    out_dir.mkdir(parents=True, exist_ok=True)
    umi_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("NT466_*.csv")):
        df = pd.read_csv(path, dtype={"ID": str})
        collapsed, annotated = collapse_umis_hd_ratio(df, max_hd=3, min_ratio=10)
        collapsed.to_csv(out_dir / f"{path.stem}__collapsed_hd3_ratio10.csv", index=False)
        annotated.to_csv(out_dir / f"{path.stem}__annotated_hd3_ratio10.csv", index=False)
        per_id = collapsed.groupby("ID", as_index=False).size().rename(columns={"size": "UMI_count"})
        per_id.to_csv(umi_dir / f"{path.stem}__collapsed_hd3_ratio10__perID_UMI_counts.csv", index=False)
        print(f"collapsed {path.name}")


def _iter_gene_tokens(values):
    for value in values:
        if pd.isna(value):
            continue
        for token in str(value).split(GENE_SEP):
            token = token.strip()
            if token:
                yield token


def _collapse_unique_strings(values):
    seen = set()
    out = []
    for token in _iter_gene_tokens(values):
        if token not in seen:
            seen.add(token)
            out.append(token)
    return GENE_SEP.join(out) if out else pd.NA


def collapse_pair_counts_from_csv(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    has_gene = "gene" in header.columns
    usecols = ["ID", "UMI", "count"] + (["gene"] if has_gene else [])
    df = pd.read_csv(path, usecols=usecols, dtype={"ID": "string", "UMI": "string", "gene": "string"})
    if has_gene:
        out = df.groupby(["ID", "UMI"], as_index=False, sort=False).agg(gene=("gene", _collapse_unique_strings), count=("count", "sum"))
    else:
        out = df.groupby(["ID", "UMI"], as_index=False, sort=False)["count"].sum()
        out["gene"] = pd.NA
    out = out.rename(columns={"ID": "barcode", "UMI": "cell_id"})
    return out[["barcode", "cell_id", "gene", "count"]]


def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.size == 0:
        return pvalues.copy()
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    n = ranked.size
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def one_sided_tail(obs, n, p, rho):
    obs = np.asarray(obs, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    if rho <= 1e-12:
        return binom.sf(obs - 1, n, p)
    kappa = max((1.0 / rho) - 1.0, 1e-6)
    return betabinom.sf(obs - 1, n, np.clip(p * kappa, 1e-12, None), np.clip((1.0 - p) * kappa, 1e-12, None))


def estimate_global_rho(df: pd.DataFrame, params: DecontamParams) -> float:
    count_col = "clean_count" if "clean_count" in df.columns else "recipient_count"
    if count_col == "clean_count" and "barcode_total" in df.columns:
        n = df["barcode_total"].to_numpy(float)
    else:
        n = df.groupby("barcode")[count_col].transform("sum").to_numpy(float)
    p = (df["donor_count"].to_numpy(float) + params.pseudocount) / (
        df["donor_total"].to_numpy(float) + params.pseudocount * df["support_n_ids"].to_numpy(float)
    )
    y = df[count_col].to_numpy(float)
    mu = n * p
    var_bin = n * p * (1 - p)
    flagged = df["flagged"].to_numpy(bool) if "flagged" in df.columns else np.zeros(len(df), dtype=bool)
    mask = (n > 1) & (var_bin > 1e-8) & (mu >= 0.5) & (df["donor_n_ids"].to_numpy(int) >= params.min_donor_ids) & (~flagged)
    if mask.sum() < 100:
        return 0.0
    q = ((y[mask] - mu[mask]) ** 2 / var_bin[mask] - 1.0) / np.maximum(n[mask] - 1.0, 1.0)
    finite = np.isfinite(q)
    if finite.sum() < 100:
        return 0.0
    q = q[finite]
    mu_mask = mu[mask][finite]
    lo, hi = np.quantile(q, [0.05, 0.95])
    keep = (q >= lo) & (q <= hi)
    if keep.sum() < 100:
        return 0.0
    weights = np.clip(mu_mask[keep], 0.5, 50.0)
    rho = np.average(q[keep], weights=weights)
    return float(np.clip(rho, 0.0, params.rho_cap))


def decontam_with_reference(donor_pairs: pd.DataFrame, recipient_pairs: pd.DataFrame, params: DecontamParams):
    donor_pairs = donor_pairs.rename(columns={"count": "donor_count"}).copy()
    recipient_pairs = recipient_pairs.rename(columns={"count": "recipient_count"}).copy()
    donor_stats = donor_pairs.groupby("barcode", as_index=False).agg(donor_total=("donor_count", "sum"), donor_n_ids=("cell_id", "nunique"))
    work = recipient_pairs.merge(donor_pairs[["barcode", "cell_id", "donor_count"]], on=["barcode", "cell_id"], how="left")
    work["donor_count"] = work["donor_count"].fillna(0).astype(int)
    work = work.merge(donor_stats, on="barcode", how="left")
    work[["donor_total", "donor_n_ids"]] = work[["donor_total", "donor_n_ids"]].fillna(0).astype(int)
    recipient_only = work.assign(recipient_only=work["donor_count"].eq(0)).groupby("barcode", as_index=False).agg(recipient_only_n_ids=("recipient_only", "sum"))
    work = work.merge(recipient_only, on="barcode", how="left")
    work["recipient_only_n_ids"] = work["recipient_only_n_ids"].fillna(0).astype(int)
    work["support_n_ids"] = work["donor_n_ids"] + work["recipient_only_n_ids"]
    work["clean_count"] = work["recipient_count"].astype(int)
    work["hard_cutoff_flag"] = False
    if params.hard_count_cutoff is not None:
        work["hard_cutoff_flag"] = work["recipient_count"].to_numpy(int) > params.hard_count_cutoff
        work.loc[work["hard_cutoff_flag"], "clean_count"] = 0
    work["stat_flagged"] = False
    work["flagged"] = work["hard_cutoff_flag"].copy()
    work["p_value"] = 1.0
    work["q_value"] = 1.0
    work["expected_count"] = 0.0
    work["enrichment"] = 1.0
    work["last_flag_iter"] = 0
    rho = estimate_global_rho(work, params)
    for iteration in range(1, params.max_iter + 1):
        barcode_total = work.groupby("barcode")["clean_count"].transform("sum").astype(int)
        p = (work["donor_count"].to_numpy(float) + params.pseudocount) / (
            work["donor_total"].to_numpy(float) + params.pseudocount * work["support_n_ids"].to_numpy(float)
        )
        expected = barcode_total.to_numpy(float) * p
        enrichment = (work["clean_count"].to_numpy(float) + params.pseudocount) / (expected + params.pseudocount)
        eligible = (barcode_total.to_numpy(int) >= params.min_barcode_total) & (work["donor_n_ids"].to_numpy(int) >= params.min_donor_ids) & (work["donor_total"].to_numpy(int) > 0)
        candidate = (
            ~work["hard_cutoff_flag"].to_numpy(bool)
            & eligible
            & (work["clean_count"].to_numpy(int) >= params.min_recipient_count)
            & (work["clean_count"].to_numpy(float) - expected >= params.min_excess)
            & (enrichment >= params.min_enrichment)
        )
        pvals = np.ones(len(work), dtype=float)
        qvals = np.ones(len(work), dtype=float)
        if candidate.any():
            pvals[candidate] = one_sided_tail(work.loc[candidate, "clean_count"].to_numpy(int), barcode_total[candidate].to_numpy(int), p[candidate], rho)
            qvals[candidate] = benjamini_hochberg(pvals[candidate])
        stat_now = candidate & (qvals <= params.fdr)
        new_clean = work["clean_count"].to_numpy(int).copy()
        if stat_now.any():
            new_clean[stat_now] = 0
        work["expected_count"] = expected
        work["enrichment"] = enrichment
        work["p_value"] = pvals
        work["q_value"] = qvals
        work["stat_flagged"] = work["stat_flagged"] | stat_now
        work["flagged"] = work["hard_cutoff_flag"] | work["stat_flagged"]
        work.loc[stat_now, "last_flag_iter"] = iteration
        changed = np.any(new_clean != work["clean_count"].to_numpy(int))
        work["clean_count"] = new_clean
        rho = estimate_global_rho(work, params)
        if params.verbose:
            print(f"iteration {iteration}: newly flagged={int(stat_now.sum()):,}, rho={rho:.4f}")
        if not changed:
            break
    work["removed_count"] = work["recipient_count"] - work["clean_count"]
    clean_pairs = (
        work.loc[work["clean_count"] > 0, ["barcode", "cell_id", "gene", "clean_count"]]
        .rename(columns={"barcode": "ID", "cell_id": "UMI", "clean_count": "count"})
    )
    barcode_summary = work.groupby("barcode", as_index=False).agg(
        recipient_total_raw=("recipient_count", "sum"),
        recipient_total_clean=("clean_count", "sum"),
        removed_total=("removed_count", "sum"),
        flagged_ids=("flagged", "sum"),
        recipient_n_ids=("cell_id", "nunique"),
    )
    return work, barcode_summary, clean_pairs, rho


def run_decontamination() -> None:
    out_dir = UPSTREAM_DIR / "decontamination"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = DecontamParams()
    donor = collapse_pair_counts_from_csv(UPSTREAM_DIR / "id_umi_counts_collapsed" / "NT466_K__collapsed_hd3_ratio10.csv")
    rows = []
    for name in ["NT466_B", "NT466_C", "NT466_D"]:
        input_path = UPSTREAM_DIR / "id_umi_counts_collapsed" / f"{name}__collapsed_hd3_ratio10.csv"
        recipient = collapse_pair_counts_from_csv(input_path)
        diagnostics, summary, clean, rho = decontam_with_reference(donor, recipient, params)
        sample_dir = out_dir / f"{name}__collapsed_hd3_ratio10"
        sample_dir.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(sample_dir / "recipient_id_level_decontam_diagnostics.csv", index=False)
        summary.to_csv(sample_dir / "barcode_level_decontam_summary.csv", index=False)
        clean.to_csv(sample_dir / "recipient_pairs_cleaned.csv", index=False)
        rows.append({"recipient_label": f"{name}__collapsed_hd3_ratio10", "rho_hat": rho, "clean_recipient_pairs": len(clean), "removed_counts": int(diagnostics["removed_count"].sum())})
    pd.DataFrame(rows).to_csv(out_dir / "batch_run_summary.csv", index=False)


def build_mageck_count_table() -> None:
    samples = [
        ("K", UPSTREAM_DIR / "id_umi_counts_collapsed" / "NT466_K__collapsed_hd3_ratio10.csv"),
        ("B", UPSTREAM_DIR / "decontamination" / "NT466_B__collapsed_hd3_ratio10" / "recipient_pairs_cleaned.csv"),
        ("C", UPSTREAM_DIR / "decontamination" / "NT466_C__collapsed_hd3_ratio10" / "recipient_pairs_cleaned.csv"),
        ("D", UPSTREAM_DIR / "decontamination" / "NT466_D__collapsed_hd3_ratio10" / "recipient_pairs_cleaned.csv"),
    ]
    count_tables = []
    gene_tables = []
    for sample, path in samples:
        df = pd.read_csv(path, usecols=["ID", "gene", "count"], dtype={"ID": "string", "gene": "string"})
        counts = df.groupby("ID", as_index=False, sort=False)["count"].sum().rename(columns={"ID": "sgRNA", "count": sample})
        gene_map = df.loc[df["gene"].notna(), ["ID", "gene"]].drop_duplicates().rename(columns={"ID": "sgRNA", "gene": "Gene"})
        gene_nunique = gene_map.groupby("sgRNA")["Gene"].nunique()
        bad = gene_nunique[gene_nunique > 1]
        if len(bad):
            raise ValueError(f"{path} has IDs with conflicting gene annotations")
        count_tables.append(counts)
        gene_tables.append(gene_map.drop_duplicates("sgRNA"))
    merged = count_tables[0]
    for counts in count_tables[1:]:
        merged = merged.merge(counts, on="sgRNA", how="outer")
    sample_names = [s for s, _ in samples]
    merged[sample_names] = merged[sample_names].fillna(0).astype("int64")
    genes = pd.concat(gene_tables, ignore_index=True).drop_duplicates()
    gene_nunique = genes.groupby("sgRNA")["Gene"].nunique()
    if (gene_nunique > 1).any():
        raise ValueError("Conflicting gene annotations across samples")
    genes = genes.drop_duplicates("sgRNA")
    out = merged.merge(genes, on="sgRNA", how="left")[["sgRNA", "Gene", *sample_names]]
    out["Gene"] = out["Gene"].fillna("NA")
    out = out.sort_values("sgRNA", kind="stable").reset_index(drop=True)
    mageck_dir = UPSTREAM_DIR / "mageck"
    mageck_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(mageck_dir / "NT466.count.txt", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild pre-MAGeCK Figure 5 workflow from raw FASTQ files.")
    parser.add_argument("--start-at", choices=["fastq", "merge", "collapse", "decontam", "mageck-count"], default="fastq")
    args = parser.parse_args()
    steps = ["fastq", "merge", "collapse", "decontam", "mageck-count"]
    start = steps.index(args.start_at)
    t0 = time.time()
    if start <= 0:
        build_id_umi_counts()
    if start <= 1:
        merge_count_tables()
    if start <= 2:
        collapse_merged_counts()
    if start <= 3:
        run_decontamination()
    if start <= 4:
        build_mageck_count_table()
    print(f"Upstream workflow complete in {time.time() - t0:.1f} seconds.")
    print("Run run_mageck_commands.sh next in an environment with MAGeCK installed.")


if __name__ == "__main__":
    main()
