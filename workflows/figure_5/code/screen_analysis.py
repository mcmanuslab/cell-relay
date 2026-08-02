#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/figure5_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCREEN_DIR = ROOT / "data" / "upstream" / "screen_analysis"
HITS_DIR = SCREEN_DIR / "hits"
CARD_DIR = ROOT / "outputs" / "figures" / "program_cards"
FIGURE_TABLE_DIR = ROOT / "data" / "figure_tables"
SUMMARY_DIR = ROOT / "outputs" / "summaries"
ANNOTATION_SNAPSHOT = HITS_DIR / "mygene_annotation_snapshot.json"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


FILTER = {
    "direction": None,
    "min_absz": None,
    "max_mle_fdr": None,
    "max_rra_fdr": 0.10,
    "require_both_donor_lfc": True,
    "max_neg_lfc": -0.25,
    "min_pos_lfc": 0.5,
    "drop_ntc": True,
}

TIERS = [
    ("T1-core", None, None, None, 0.05),
    ("T2-nearmiss-RRA", None, None, 0.05, 0.10),
]

ALIAS_MAP = {
    "C17orf72": "HROB",
    "C9orf170": "LINC02872",
    "DGCR14": "ESS2",
    "IL8": "CXCL8",
    "KIAA0947": "ICE1",
    "KIAA1199": "CEMIP",
    "RFWD2": "COP1",
    "SEPN1": "SELENON",
    "TSSC1": "EIPR1",
    "WRB": "GET1",
}

MYGENE_SCOPES = "symbol,alias"
MYGENE_FIELDS = "symbol,name,go,alias,taxid,entrezgene,type_of_gene"

RULES = [
    ("Polarized exocytosis & fusion", ["exocyst"]),
    ("Antigen processing & presentation (MHC-I)", ["antigen process", "antigen present", "mhc class i", r"\btapasin\b", "immunoproteasome", "peptide loading", r"\btap[12]?\b.*transport"]),
    ("TRAPP tethering complex", [r"\btrapp\b", "trafficking protein particle"]),
    ("COP9 signalosome / neddylation", ["cop9 signalosome", "neddylat", "deneddylat", r"\bnedd8\b"]),
    ("GPI-anchor & glycan surface display", ["gpi[ -]anchor", "glycosylphosphatidylinositol", "n-glycan", "n-linked glycosyl", "dolich", "mannosyltransfer", "gdp-mannose", "glycoprotein biosynthetic", "oligosaccharyltransfer"]),
    ("Ubiquitin–proteasome / protein degradation", ["ubiquitin ligase", "e3 ubiquitin", "ubiquitin-protein transferase", "deubiquitinat", "proteasom", "ubiquitin-dependent protein catab", "polyubiquitination"]),
    ("Membrane–cortex mechanics", [r"band 4\.1", "spectrin", "ezrin|radixin|moesin", r"\berm\b"]),
    ("Endosomal pH & recycling (identity)", [r"\bnhe\b", "sodium.*proton antiport", "solute carrier family 9"]),
    ("Cell polarity", ["cell polarity", "establishment or maintenance of cell polarity"]),
    ("ER-Golgi / COPII vesicular transport", ["copii", r"\bcopi\b", "endoplasmic reticulum to golgi", "intra-golgi", "golgi to endoplasmic reticulum", "retrograde vesicle.*golgi", "golgi organization", "vesicle tethering", "vesicle coating", r"\ber to golgi\b"]),
    ("Polarized exocytosis & fusion", ["exocyst", r"\bexocytos", "golgi to plasma membrane", "post-golgi vesicle", "snap receptor", r"\bsnare\b", "regulated exocytos", "vesicle docking", "neurotransmitter secretion", "synaptic vesicle", "membrane fusion"]),
    ("Autophagy–secretion & DAMP", [r"\bautophagosome\b", "mitophagy", "optineurin", "macroautophagy"]),
    ("Membrane-protein biogenesis (ER/GET/ERAD)", [r"\berad\b", "tail-anchored", r"\bget\b pathway", "signal peptidase", "protein insertion into.*membrane", "endoplasmic reticulum.*quality control"]),
    ("Endosomal pH & recycling (endo)", ["endosome to lysosome", "endosomal acidif", "endosomal ph", "early endosome.*(recycl|sort)", "rab interacting lysosomal"]),
    ("Membrane lipid remodeling", ["phosphatidylserine", "phospholipid translocat", "flippase", "phosphoinositide", "phosphatidylinositol.*(phosphat|kinase)", "sphingolipid", "myotubularin"]),
    ("PS-recognition & engulfment", ["apoptotic cell (clear|engulf|recogni)", "efferocytos", "engulfment of", "phosphatidylserine receptor"]),
    ("Surface sensing & adhesion", ["cell-cell adhesion", "cell adhesion molecul", "homophilic cell adhesion", "mechanosensit", "detection of mechanical"]),
    ("Transport — ion channels & solute carriers", ["ion channel", "potassium ion transmembrane", "voltage-gated (potassium|sodium|calcium)", "sodium ion transmembrane", "calcium ion transmembrane", "solute carrier", "aquaporin", "water transport", "atp binding cassette"]),
    ("Actin & Rho signaling", ["actin filament", "actin cytoskeleton organization", "cortical actin", "rho protein signal", "rho guanine nucleotide", "rac family", r"\bcdc42\b", "lamellipod", "filopod", "small gtpase-mediated"]),
    ("Cytokine / GPCR / surface receptor signaling", ["cytokine", "chemokine", "g protein-coupled receptor", "jak-stat", r"\bjak\b", "interleukin", "growth factor.*signal", "wnt signal", "arrestin", "activin receptor", "thrombopoietin", "mapk cascade"]),
    ("Spliceosome / RNA processing (likely general)", ["spliceosom", "mrna splic", "pre-mrna", "rna splic", r"\bsnrnp\b", "mrna 3.-end", "rna processing"]),
    ("Transcription / chromatin (likely general)", ["transcription by rna pol", "dna-templated transcription", "regulation of transcription", "chromatin", "histone", "nucleosome", "transcription factor activity", "rna polymerase ii"]),
    ("Metabolism / mitochondria (likely general)", ["mitochondrial", "oxidative phosphoryl", "respiratory chain", "tricarboxylic acid", "glycolytic", "fatty acid.*oxidation", "aerobic respiration", r"\bnadh\b"]),
]

GENERAL_PROGRAMS = {
    "Spliceosome / RNA processing (likely general)",
    "Transcription / chromatin (likely general)",
    "Metabolism / mitochondria (likely general)",
    "Other / unclear",
}

CARD_PARAMS = {
    "target_tier": "T1-core",
    "title_wrap": 36,
    "title_fontsize": 18,
    "max_cols": 2,
    "card_width": 3.2,
    "card_height": 1.9,
    "outer_pad_x": 0.20,
    "outer_pad_y": 0.18,
    "label_width_frac": 0.78,
    "label_height_frac": 0.44,
    "label_facecolor": "#FFFFFF",
    "label_edgecolor": "black",
    "card_edgecolor": "black",
    "card_linewidth": 1.7,
    "gene_fontsize": 17,
    "title_space": 0.95,
    "figure_facecolor": "white",
    "missing_color": "#D9D9D9",
    "rra_maxfdr_colors": ["#a68ff8", "#FFFFFF"],
    "rra_maxfdr_limits": [0.0, 0.05],
    "lfc_colors": ["#5d8af7", "#ffffff", "#ed8590"],
    "lfc_limits": [-2.5, 2.5],
    "lfc_center": 0.0,
    "save_dpi": 300,
}

EXPECTED_TIER_COUNTS = {"T1-core": 168, "T2-nearmiss-RRA": 38}
EXPECTED_PROGRAM_COUNTS = {
    "Other / unclear": 51,
    "Transcription / chromatin (likely general)": 22,
    "Spliceosome / RNA processing (likely general)": 21,
    "ER-Golgi / COPII vesicular transport": 15,
    "Ubiquitin–proteasome / protein degradation": 15,
    "Polarized exocytosis & fusion": 12,
    "Cytokine / GPCR / surface receptor signaling": 10,
    "Transport — ion channels & solute carriers": 9,
    "TRAPP tethering complex": 6,
    "Actin & Rho signaling": 6,
    "Antigen processing & presentation (MHC-I)": 5,
    "GPI-anchor & glycan surface display": 5,
    "Membrane lipid remodeling": 5,
    "COP9 signalosome / neddylation": 5,
    "Metabolism / mitochondria (likely general)": 4,
    "Membrane-protein biogenesis (ER/GET/ERAD)": 4,
    "Surface sensing & adhesion": 3,
    "Autophagy–secretion & DAMP": 3,
    "Cell polarity": 2,
    "Endosomal pH & recycling": 2,
    "Membrane–cortex mechanics": 1,
}
EXPECTED_T1_PROGRAM_COUNTS = {
    "ER-Golgi / COPII vesicular transport": 14,
    "Spliceosome / RNA processing (likely general)": 16,
    "TRAPP tethering complex": 6,
    "Transcription / chromatin (likely general)": 16,
    "Ubiquitin–proteasome / protein degradation": 14,
    "Cytokine / GPCR / surface receptor signaling": 8,
    "Other / unclear": 36,
    "Antigen processing & presentation (MHC-I)": 5,
    "Actin & Rho signaling": 5,
    "Membrane-protein biogenesis (ER/GET/ERAD)": 4,
    "Metabolism / mitochondria (likely general)": 4,
    "Polarized exocytosis & fusion": 10,
    "Cell polarity": 1,
    "COP9 signalosome / neddylation": 4,
    "Transport — ion channels & solute carriers": 9,
    "Membrane lipid remodeling": 5,
    "GPI-anchor & glycan surface display": 5,
    "Endosomal pH & recycling": 2,
    "Surface sensing & adhesion": 2,
    "Autophagy–secretion & DAMP": 2,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _normalized_direction(filters: dict):
    direction = filters.get("direction")
    if direction is None:
        return None
    if isinstance(direction, str):
        direction = direction.strip().lower()
        if direction in {"", "none", "both"}:
            return None
    if direction not in {"depletion", "enrichment"}:
        raise ValueError('FILTER["direction"] must be "depletion", "enrichment", or None')
    return direction


def _add_directional_rra_fdr(table: pd.DataFrame) -> pd.DataFrame:
    required = ["B_lfc", "C_lfc", "B_neg_fdr", "C_neg_fdr", "B_pos_fdr", "C_pos_fdr"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise KeyError(f"Missing required RRA columns: {missing}")
    table = table.copy()
    table["B_dir_fdr"] = table["B_pos_fdr"].where(table["B_lfc"] > 0, table["B_neg_fdr"])
    table["C_dir_fdr"] = table["C_pos_fdr"].where(table["C_lfc"] > 0, table["C_neg_fdr"])
    table["rra_maxfdr"] = table[["B_dir_fdr", "C_dir_fdr"]].max(axis=1)
    return table


def _donor_lfc_gate(table: pd.DataFrame, filters: dict) -> pd.Series:
    direction = _normalized_direction(filters)
    max_neg_lfc = filters.get("max_neg_lfc")
    min_pos_lfc = filters.get("min_pos_lfc")
    neg_b = table["B_lfc"] <= max_neg_lfc if max_neg_lfc is not None else table["B_lfc"] < 0
    neg_c = table["C_lfc"] <= max_neg_lfc if max_neg_lfc is not None else table["C_lfc"] < 0
    pos_b = table["B_lfc"] >= min_pos_lfc if min_pos_lfc is not None else table["B_lfc"] > 0
    pos_c = table["C_lfc"] >= min_pos_lfc if min_pos_lfc is not None else table["C_lfc"] > 0
    if direction == "depletion":
        return neg_b & neg_c
    if direction == "enrichment":
        return pos_b & pos_c
    return (neg_b & neg_c) | (pos_b & pos_c)


def _passes_filter(table: pd.DataFrame, filters: dict) -> pd.Series:
    direction = _normalized_direction(filters)
    keep = pd.Series(True, index=table.index)
    if direction == "depletion":
        keep &= table["MLE_beta"] < 0
    elif direction == "enrichment":
        keep &= table["MLE_beta"] > 0
    if filters.get("require_both_donor_lfc"):
        keep &= _donor_lfc_gate(table, filters)
    if filters.get("min_absz") is not None:
        keep &= table["absz"] >= filters["min_absz"]
    if filters.get("max_mle_fdr") is not None:
        keep &= table["MLE_fdr"] <= filters["max_mle_fdr"]
    if filters.get("max_rra_fdr") is not None:
        keep &= table["rra_maxfdr"] <= filters["max_rra_fdr"]
    if filters.get("drop_ntc"):
        keep &= ~table["Gene"].astype(str).str.startswith("NTC")
    return keep


def _assign_tier(row: pd.Series) -> str:
    for name, zmin, zmax, fmin, fmax in TIERS:
        if zmin is not None and row["absz"] < zmin:
            continue
        if zmax is not None and row["absz"] > zmax:
            continue
        if fmin is not None and row["rra_maxfdr"] < fmin:
            continue
        if fmax is not None and row["rra_maxfdr"] > fmax:
            continue
        return name
    return "untiered"


def call_hits(master: pd.DataFrame) -> pd.DataFrame:
    table = master.copy()
    table["absz"] = table["MLE_z"].abs()
    genes = table.sort_values("absz", ascending=False).drop_duplicates("Gene").reset_index(drop=True)
    genes = _add_directional_rra_fdr(genes)
    hits = genes.loc[_passes_filter(genes, FILTER)].copy()
    hits["origin"] = "new"
    hits["tier"] = hits.apply(_assign_tier, axis=1)
    hits = hits[hits["tier"] != "untiered"].copy()
    columns = ["Gene", "MLE_beta", "MLE_z", "absz", "MLE_fdr", "rra_maxfdr", "B_lfc", "C_lfc", "tier", "origin"]
    if "DepMap" in hits.columns:
        columns.append("DepMap")
    if "Surfaceome" in hits.columns:
        hits["surface"] = hits["Surfaceome"].astype(str).str.lower().eq("surface")
        columns.append("surface")
    return hits[columns].reset_index(drop=True)


def _clean_program(program: str) -> str:
    return program.replace(" (identity)", "").replace(" (endo)", "")


def classify(annotation_blob: str) -> tuple[str, str]:
    for program, patterns in RULES:
        for pattern in patterns:
            if re.search(pattern, annotation_blob):
                return _clean_program(program), pattern
    return "Other / unclear", "no keyword match"


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _aliases(record: dict) -> list[str]:
    return [value for value in _as_list(record.get("alias")) if isinstance(value, str)]


def _annotation_blob(record: dict) -> str:
    def terms(category: str) -> list[str]:
        values = record.get("go", {}).get(category, [])
        values = [values] if isinstance(values, dict) else values
        return [term.get("term", "") for term in values if isinstance(term, dict)]

    return " | ".join([record.get("name", "")] + terms("BP") + terms("MF")).lower()


def _record_rank(record: dict, blob: str, gene: str, query_term: str, preferred_symbol: str) -> tuple:
    name = str(record.get("name", "")).lower()
    symbol = str(record.get("symbol", ""))
    aliases = {alias.upper() for alias in _aliases(record)}
    try:
        score = float(record.get("_score", 0) or 0)
    except (TypeError, ValueError):
        score = 0.0
    return (
        record.get("taxid") == 9606,
        symbol.upper() == preferred_symbol.upper(),
        symbol.upper() == str(query_term).upper(),
        symbol.upper() == str(gene).upper(),
        str(gene).upper() in aliases or str(query_term).upper() in aliases,
        "ubiquitin ligase" in name,
        str(record.get("type_of_gene", "")).lower() == "protein-coding",
        len(blob),
        score,
        str(record.get("_id", "")),
    )


def fetch_annotation_snapshot(genes: list[str]) -> dict:
    try:
        import mygene
    except ImportError as exc:
        raise RuntimeError("Refreshing annotations requires the `mygene` package.") from exc

    preferred = {gene: ALIAS_MAP.get(gene, gene) for gene in genes}
    term_to_genes: dict[str, set[str]] = {}
    for gene in genes:
        for term in {gene, preferred[gene]}:
            term_to_genes.setdefault(str(term), set()).add(gene)
            term_to_genes.setdefault(str(term).upper(), set()).add(gene)
    query_terms = sorted({str(term) for gene in genes for term in (gene, preferred[gene])})

    result = mygene.MyGeneInfo().querymany(
        query_terms,
        scopes=MYGENE_SCOPES,
        fields=MYGENE_FIELDS,
        species="human",
        returnall=True,
        verbose=False,
    )
    if not isinstance(result, dict) or "out" not in result:
        raise RuntimeError("Unexpected MyGene querymany response; expected an `out` list.")

    selected: dict[str, dict] = {}
    ranks: dict[str, tuple] = {}
    for record in result["out"]:
        query = record.get("query")
        if not query or record.get("notfound"):
            continue
        candidates = term_to_genes.get(str(query)) or term_to_genes.get(str(query).upper()) or set()
        blob = _annotation_blob(record)
        for gene in candidates:
            rank = _record_rank(record, blob, gene, str(query), preferred[gene])
            if gene not in selected or rank > ranks[gene]:
                selected[gene] = {
                    "preferred_symbol": preferred[gene],
                    "selected_query": query,
                    "annotation_blob": blob,
                    "record": record,
                }
                ranks[gene] = rank

    missing = [gene for gene in genes if gene not in selected or not selected[gene]["annotation_blob"].strip(" |")]
    if missing:
        details = ", ".join(f"{gene}->{preferred[gene]}" for gene in missing)
        raise RuntimeError(f"MyGene returned no name/GO annotation for: {details}")

    snapshot = {
        "schema_version": 1,
        "retrieved_at_utc": utc_now(),
        "endpoint": "https://mygene.info/v3",
        "client": f"mygene {getattr(mygene, '__version__', 'unknown')}",
        "scopes": MYGENE_SCOPES,
        "fields": MYGENE_FIELDS,
        "species": "human",
        "genes": selected,
    }
    write_json(ANNOTATION_SNAPSHOT, snapshot)
    return snapshot


def load_annotation_snapshot(genes: list[str], refresh: bool = False) -> dict:
    snapshot = fetch_annotation_snapshot(genes) if refresh else None
    if snapshot is None:
        if not ANNOTATION_SNAPSHOT.exists():
            raise FileNotFoundError(
                f"Missing {ANNOTATION_SNAPSHOT}. Run with --refresh-mygene once to create it."
            )
        snapshot = json.loads(ANNOTATION_SNAPSHOT.read_text())
    cached = set(snapshot.get("genes", {}))
    requested = set(genes)
    if cached != requested:
        missing = sorted(requested - cached)
        extra = sorted(cached - requested)
        raise RuntimeError(
            "The MyGene snapshot does not match the current hit set. "
            f"Missing={missing[:10]}, extra={extra[:10]}. Refresh it with --refresh-mygene."
        )
    return snapshot


def build_relay_hits(master: pd.DataFrame, refresh_mygene: bool = False) -> tuple[pd.DataFrame, dict]:
    hits = call_hits(master)
    snapshot = load_annotation_snapshot(hits["Gene"].astype(str).tolist(), refresh=refresh_mygene)
    programs = []
    evidence = []
    evidence_rows = []
    for gene in hits["Gene"].astype(str):
        cached = snapshot["genes"][gene]
        program, matched = classify(cached["annotation_blob"])
        record = cached["record"]
        programs.append(program)
        evidence.append(matched)
        evidence_rows.append(
            {
                "Gene": gene,
                "preferred_symbol": cached["preferred_symbol"],
                "selected_query": cached["selected_query"],
                "resolved_symbol": record.get("symbol"),
                "entrezgene": record.get("entrezgene"),
                "gene_name": record.get("name"),
                "program": program,
                "program_evidence": matched,
                "annotation_blob": cached["annotation_blob"],
            }
        )
    hits["program"] = programs
    hits["program_evidence"] = evidence
    hits["likely_general"] = hits["program"].isin(GENERAL_PROGRAMS)
    hits = hits.sort_values(["tier", "MLE_z"]).reset_index(drop=True)

    HITS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    hits.to_csv(HITS_DIR / "relay_hits_regenerated.csv", index=False)
    hits.to_csv(FIGURE_TABLE_DIR / "relay_hits_regenerated.csv", index=False)
    pd.DataFrame(evidence_rows).to_csv(FIGURE_TABLE_DIR / "relay_hit_program_annotations.csv", index=False)

    summary = {
        "rows": int(len(hits)),
        "tiers": {str(key): int(value) for key, value in hits["tier"].value_counts().items()},
        "programs": int(hits["program"].nunique()),
        "relevant": int((~hits["likely_general"]).sum()),
        "likely_general": int(hits["likely_general"].sum()),
        "program_distribution": {str(key): int(value) for key, value in hits["program"].value_counts().items()},
        "annotation_snapshot": str(ANNOTATION_SNAPSHOT.relative_to(ROOT)),
        "annotation_retrieved_at_utc": snapshot.get("retrieved_at_utc"),
        "annotation_refreshed": bool(refresh_mygene),
    }
    return hits, summary


def _color_objects() -> dict:
    rra_min, rra_max = CARD_PARAMS["rra_maxfdr_limits"]
    lfc_min, lfc_max = CARD_PARAMS["lfc_limits"]
    return {
        "rra_cmap": LinearSegmentedColormap.from_list("rra_maxfdr_cmap", CARD_PARAMS["rra_maxfdr_colors"]),
        "lfc_cmap": LinearSegmentedColormap.from_list("lfc_cmap", CARD_PARAMS["lfc_colors"]),
        "rra_norm": Normalize(vmin=rra_min, vmax=rra_max, clip=True),
        "lfc_norm": TwoSlopeNorm(vmin=lfc_min, vcenter=CARD_PARAMS["lfc_center"], vmax=lfc_max),
    }


def _value_color(value, cmap, norm):
    return CARD_PARAMS["missing_color"] if pd.isna(value) else cmap(norm(float(value)))


def _gene_fontsize(gene: str) -> float:
    length = len(str(gene))
    if length <= 7:
        return CARD_PARAMS["gene_fontsize"]
    if length <= 10:
        return CARD_PARAMS["gene_fontsize"] - 2
    return max(9, CARD_PARAMS["gene_fontsize"] - 2 - 0.55 * (length - 10))


def _draw_card(ax, x: float, y: float, row: pd.Series, colors: dict) -> None:
    width = CARD_PARAMS["card_width"]
    height = CARD_PARAMS["card_height"]
    edge_width = CARD_PARAMS["card_linewidth"]
    ax.add_patch(Rectangle((x, y), width, height, facecolor=_value_color(row["rra_maxfdr"], colors["rra_cmap"], colors["rra_norm"]), edgecolor=CARD_PARAMS["card_edgecolor"], linewidth=edge_width, joinstyle="miter", zorder=1))
    inner_x = x + CARD_PARAMS["outer_pad_x"]
    inner_y = y + CARD_PARAMS["outer_pad_y"]
    inner_w = width - 2 * CARD_PARAMS["outer_pad_x"]
    inner_h = height - 2 * CARD_PARAMS["outer_pad_y"]
    ax.add_patch(Rectangle((inner_x, inner_y), inner_w / 2, inner_h, facecolor=_value_color(row["B_lfc"], colors["lfc_cmap"], colors["lfc_norm"]), edgecolor=CARD_PARAMS["card_edgecolor"], linewidth=edge_width * 0.9, joinstyle="miter", zorder=2))
    ax.add_patch(Rectangle((inner_x + inner_w / 2, inner_y), inner_w / 2, inner_h, facecolor=_value_color(row["C_lfc"], colors["lfc_cmap"], colors["lfc_norm"]), edgecolor=CARD_PARAMS["card_edgecolor"], linewidth=edge_width * 0.9, joinstyle="miter", zorder=2))
    label_w = width * CARD_PARAMS["label_width_frac"]
    label_h = height * CARD_PARAMS["label_height_frac"]
    ax.add_patch(Rectangle((x + (width - label_w) / 2, y + (height - label_h) / 2), label_w, label_h, facecolor=CARD_PARAMS["label_facecolor"], edgecolor=CARD_PARAMS["label_edgecolor"], linewidth=edge_width * 0.95, joinstyle="miter", zorder=5))
    ax.text(x + width / 2, y + height / 2, row["Gene"], ha="center", va="center", fontsize=_gene_fontsize(row["Gene"]), fontweight="bold", color="black", zorder=6)


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).replace("\n", " ")).strip("_") or "program"


def _plot_program(program: str, table: pd.DataFrame, colors: dict):
    cards = table.reset_index(drop=True)
    n_columns = min(CARD_PARAMS["max_cols"], max(1, len(cards)))
    n_rows = math.ceil(len(cards) / n_columns)
    figure_width = n_columns * CARD_PARAMS["card_width"]
    figure_height = n_rows * CARD_PARAMS["card_height"] + CARD_PARAMS["title_space"]
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=CARD_PARAMS["save_dpi"])
    fig.patch.set_facecolor(CARD_PARAMS["figure_facecolor"])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, figure_width)
    ax.set_ylim(0, figure_height)
    ax.axis("off")
    title = textwrap.fill(str(program).strip() or "Unassigned", width=CARD_PARAMS["title_wrap"])
    ax.text(figure_width / 2, n_rows * CARD_PARAMS["card_height"] + CARD_PARAMS["title_space"] * 0.56, title, ha="center", va="center", fontsize=CARD_PARAMS["title_fontsize"], color="black")
    for index, row in cards.iterrows():
        grid_row, grid_column = divmod(index, n_columns)
        x = grid_column * CARD_PARAMS["card_width"]
        y = (n_rows - 1 - grid_row) * CARD_PARAMS["card_height"]
        _draw_card(ax, x, y, row, colors)
    return fig


def _plot_color_legend(colors: dict):
    fig = plt.figure(figsize=(7.0, 3.0), dpi=CARD_PARAMS["save_dpi"])
    fig.patch.set_facecolor(CARD_PARAMS["figure_facecolor"])
    ax_rra = fig.add_axes([0.16, 0.66, 0.72, 0.12])
    rra_map = ScalarMappable(cmap=colors["rra_cmap"], norm=colors["rra_norm"])
    rra_map.set_array([])
    rra_bar = fig.colorbar(rra_map, cax=ax_rra, orientation="horizontal")
    rra_bar.set_label("Outer card color: rra_maxfdr", fontsize=11, labelpad=8)
    rra_ticks = [colors["rra_norm"].vmin, (colors["rra_norm"].vmin + colors["rra_norm"].vmax) / 2, colors["rra_norm"].vmax]
    rra_bar.set_ticks(rra_ticks)
    rra_bar.set_ticklabels([f"{value:.3g}" for value in rra_ticks])

    ax_lfc = fig.add_axes([0.16, 0.36, 0.72, 0.12])
    lfc_map = ScalarMappable(cmap=colors["lfc_cmap"], norm=colors["lfc_norm"])
    lfc_map.set_array([])
    lfc_bar = fig.colorbar(lfc_map, cax=ax_lfc, orientation="horizontal")
    lfc_bar.set_label("Inner colors: B_lfc and C_lfc", fontsize=11, labelpad=8)
    lfc_ticks = [colors["lfc_norm"].vmin, CARD_PARAMS["lfc_center"], colors["lfc_norm"].vmax]
    lfc_bar.set_ticks(lfc_ticks)
    lfc_bar.set_ticklabels([f"{value:.3g}" for value in lfc_ticks])

    ax_missing = fig.add_axes([0.16, 0.12, 0.72, 0.10])
    ax_missing.axis("off")
    ax_missing.legend(handles=[Patch(facecolor=CARD_PARAMS["missing_color"], edgecolor="black", label="Missing / NA value")], loc="center left", frameon=False, fontsize=10)
    fig.suptitle("Card color scale legend", fontsize=13, fontweight="bold", y=0.95)
    return fig


def _save_pair(fig, stem: Path, pad_inches: float = 0.03) -> None:
    for extension in ["pdf", "png"]:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=CARD_PARAMS["save_dpi"], bbox_inches="tight", pad_inches=pad_inches, facecolor=fig.get_facecolor())


def plot_program_cards(relay_hits: pd.DataFrame) -> dict:
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    cards = relay_hits.loc[relay_hits["tier"].eq(CARD_PARAMS["target_tier"]), ["program", "Gene", "B_lfc", "C_lfc", "rra_maxfdr", "tier"]].copy()
    cards["source_row"] = cards.index
    cards = cards.reset_index(drop=True)
    cards.to_csv(SUMMARY_DIR / "program_card_input_generated.csv", index=False)
    write_json(SUMMARY_DIR / "program_card_parameters_used.json", CARD_PARAMS)

    colors = _color_objects()
    legend = _plot_color_legend(colors)
    _save_pair(legend, CARD_DIR / "color_scale_legend", pad_inches=0.05)
    plt.close(legend)

    rows = []
    expected = {"color_scale_legend.pdf", "color_scale_legend.png"}
    groups = list(cards.groupby("program", sort=False))
    for index, (program, table) in enumerate(groups, start=1):
        filename = f"program_{index:02d}_{_sanitize_filename(program)}"
        fig = _plot_program(program, table, colors)
        _save_pair(fig, CARD_DIR / filename)
        plt.close(fig)
        expected.update({f"{filename}.pdf", f"{filename}.png"})
        rows.append(
            {
                "program": program,
                "gene_cards": int(len(table)),
                "pdf": relpath(CARD_DIR / f"{filename}.pdf"),
                "png": relpath(CARD_DIR / f"{filename}.png"),
            }
        )

    for path in CARD_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".pdf", ".png"} and path.name not in expected:
            path.unlink()
    pd.DataFrame(rows).to_csv(SUMMARY_DIR / "program_card_render_summary_generated.csv", index=False)
    return {
        "target_tier": CARD_PARAMS["target_tier"],
        "rows": int(len(cards)),
        "programs": int(len(groups)),
        "pdfs": int(len(list(CARD_DIR.glob("*.pdf")))),
        "pngs": int(len(list(CARD_DIR.glob("*.png")))),
        "program_distribution": {str(row["program"]): int(row["gene_cards"]) for row in rows},
    }


def run_screen_hit_workflow(refresh_mygene: bool = False) -> dict:
    master_path = SCREEN_DIR / "NT466_combined.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing screen master table: {master_path}")
    master = pd.read_csv(master_path)
    hits, hit_summary = build_relay_hits(master, refresh_mygene=refresh_mygene)
    card_summary = plot_program_cards(hits)
    summary = {
        "workflow": "Figure 5 updated hit calling and T1-core program cards",
        "run_at_utc": utc_now(),
        "python": sys.version,
        "filter": FILTER,
        "tiers": TIERS,
        "hits": hit_summary,
        "program_cards": card_summary,
    }
    validation = {
        "hit_rows_match": hit_summary["rows"] == 206,
        "tier_counts_match": hit_summary["tiers"] == EXPECTED_TIER_COUNTS,
        "program_counts_match": hit_summary["program_distribution"] == EXPECTED_PROGRAM_COUNTS,
        "t1_card_rows_match": card_summary["rows"] == 168,
        "t1_program_counts_match": card_summary["program_distribution"] == EXPECTED_T1_PROGRAM_COUNTS,
    }
    summary["source_notebook_validation"] = validation
    write_json(SUMMARY_DIR / "screen_analysis_summary.json", summary)
    if not all(validation.values()):
        raise RuntimeError(
            "Updated screen-analysis outputs do not match the counts recorded in the source notebooks; "
            f"see {SUMMARY_DIR / 'screen_analysis_summary.json'}."
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild updated Figure 5 hit calls and T1-core program cards.")
    parser.add_argument("--refresh-mygene", action="store_true", help="Query MyGene.info and replace the packaged annotation snapshot.")
    args = parser.parse_args()
    summary = run_screen_hit_workflow(refresh_mygene=args.refresh_mygene)
    print(json.dumps({"hits": summary["hits"]["rows"], "cards": summary["program_cards"]["rows"], "programs": summary["program_cards"]["programs"]}, indent=2))


if __name__ == "__main__":
    main()
