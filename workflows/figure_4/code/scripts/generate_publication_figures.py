#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUB_ROOT = PROJECT_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PUB_ROOT / "outputs" / "summaries" / "mplconfig"))
(PUB_ROOT / "outputs" / "summaries" / "mplconfig").mkdir(parents=True, exist_ok=True)

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
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import sparse
from scipy import stats
from scipy.cluster import hierarchy


ROOT = PROJECT_ROOT
PUB = PUB_ROOT
TABLES = PUB / "data" / "figure_tables"
QC = PUB / "outputs" / "summaries"
FIG_PDF = PUB / "outputs" / "figures"
FIG_PNG = PUB / "outputs" / "figures"

PALETTE = {
    "dendritic cell": "#8fb3d9",
    "OTI": "#f6b26b",
    "C57BL6": "#93c47d",
    "C57BL/6": "#93c47d",
    "donor": "#4f8fc0",
    "recipient": "#f28e5c",
    "multi": "#d7bde2",
    "no_treatment": "#b7b7b7",
    "LPS": "#f4a6a6",
    "PolyIC": "#a8d5ba",
    "IFNg": "#a9c4f5",
}
OTI_BLUE = "#b4d9f2"
C57_GREY = "#a9a29d"
CONTROL_GREY = "#7a716b"
SKY_BLUE = "#84c7ff"

PEPTIDE_COLORS = {
    "N4": "#8fb3d9",
    "A2": "#f6b26b",
    "Y3": "#93c47d",
    "Q4": "#f4a6a6",
    "T4": "#a8d5ba",
    "V4": "#a9c4f5",
    "G4": "#d7bde2",
    "D4": "#cfcfcf",
    "Q4R7": "#c49a6c",
    "Q4H7": "#80c7c5",
    "Q7": "#e5c07b",
    "E1": "#b8a9e6",
    "CATNB": "#d9a5b3",
    "LCMV": "#9cc985",
    "TB": "#ddb87c",
    "MCMV": "#9fb5d9",
}

TREATMENT_ORDER = ["no_treatment", "LPS", "PolyIC", "IFNg"]
T_CELL_TYPES = ["OTI", "C57BL6"]
CONTROL_PEPTIDES = ["CATNB", "LCMV", "TB", "MCMV"]
KEY_T_CELL_SIGNATURES = ["Proliferation", "Naive_memory", "IFN_stress", "Cytotoxic_effector"]
PEPTIDE_TRANSFER_UMI_MIN_EXCLUSIVE = 2
PEPTIDE_TRANSFER_FRACTION_MIN = 0.90
PEPTIDE_TRANSFER_SCALE = 10000.0
TREATMENT_COLORS = {
    "no_treatment": "#b7b7b7",
    "LPS": "#f4a6a6",
    "PolyIC": "#a8d5ba",
    "IFNg": "#a9c4f5",
}
HEATMAP_LOW = "#5d8af7"
HEATMAP_MID = "#ffffff"
HEATMAP_HIGH = "#ed8590"

DC_PROGRAMS = {
    "costim_checkpoint": ["Cd40", "Cd70", "Cd80", "Cd83", "Cd86", "Cd274", "Pdcd1lg2", "Icosl", "Tnfsf4", "Tnfsf9", "Tnfsf18", "Havcr2", "Lgals9", "Relb", "Nfkb1", "Nfkb2"],
    "antigen_presentation": ["H2-Aa", "H2-Ab1", "H2-Eb1", "H2-DMa", "H2-DMb1", "H2-DMb2", "H2-Oa", "H2-Ob", "Cd74", "H2-K1", "H2-D1", "B2m", "Tap1", "Tap2", "Tapbp", "Psmb8", "Psmb9", "Psmb10", "Ciita", "Nlrc5", "Ctss", "Ctsb", "Ctsc", "Ctsd", "Ctsl", "Lamp1", "Lamp2"],
    "cytokines_inflammatory": ["Il12a", "Il12b", "Il23a", "Il27", "Ebi3", "Il6", "Il1a", "Il1b", "Tnf", "Tnfsf15", "Il10", "Il18", "Ifnb1", "Csf2", "Csf3", "Nos2", "Ptgs2"],
    "chemokines_migration": ["Ccl2", "Ccl3", "Ccl4", "Ccl5", "Ccl7", "Ccl8", "Ccl17", "Ccl19", "Ccl20", "Ccl22", "Cxcl1", "Cxcl2", "Cxcl3", "Cxcl9", "Cxcl10", "Cxcl11", "Cxcl16", "Ccr7", "Xcr1", "Fscn1"],
    "ifn_response": ["Stat1", "Stat2", "Irf1", "Irf3", "Irf5", "Irf7", "Irf8", "Irf9", "Isg15", "Ifit1", "Ifit2", "Ifit3", "Ifitm1", "Ifitm2", "Ifitm3", "Oas1a", "Oas1g", "Oas2", "Oas3", "Mx1", "Mx2", "Rsad2", "Rigi", "Ifih1", "Usp18", "Gbp2", "Gbp3", "Gbp4", "Gbp5"],
    "prr_nfkb": ["Tlr2", "Tlr3", "Tlr4", "Tlr7", "Tlr9", "Cd14", "Ly96", "Myd88", "Ticam1", "Traf6", "Map3k8", "Nlrp3", "Nod1", "Nod2", "Rela", "Rel", "Nfkbia", "Nfkbiz", "Tnfaip3", "Jun", "Junb", "Fos", "Fosb", "Atf3", "Dusp1", "Dusp2"],
    "stress_myeloid": ["Lcn2", "Slpi", "S100a8", "S100a9", "Sod2", "Hmox1", "Lyz2", "Lpl", "Srgn", "Apoe", "Trem1", "Trem2", "Tyrobp", "Lgals3", "Axl", "Mertk", "Socs1", "Socs3", "Birc2", "Birc3", "Serpinb9"],
    "dc_context": ["Itgax", "Flt3", "Zbtb46", "Batf3", "Clec9a", "Clec10a", "Cd209a", "Sirpa", "Itgae", "Ly75", "Ccr2", "Csf1r"],
}

PRIMARY_DC_DGE_GENES = [
    ("costim_checkpoint", "Cd80", "B7-1 costimulation; LPS-up in current DGE."),
    ("costim_checkpoint", "Cd86", "B7-2 costimulation; induced in both contrasts."),
    ("costim_checkpoint", "Cd274", "PD-L1 checkpoint ligand; treatment-responsive checkpoint marker."),
    ("antigen_presentation", "Cd74", "MHC-II invariant chain; strongest IFNg antigen-presentation signal."),
    ("antigen_presentation", "H2-Aa", "MHC-II alpha chain; IFNg-up antigen-presentation anchor."),
    ("antigen_presentation", "H2-Ab1", "MHC-II beta chain; IFNg-up antigen-presentation anchor."),
    ("antigen_presentation", "Ciita", "MHC-II transcriptional regulator; IFNg-up mechanism marker."),
    ("cytokines_inflammatory", "Nos2", "Inflammatory effector enzyme; LPS-up and modest IFNg-up."),
    ("cytokines_inflammatory", "Tnf", "Canonical inflammatory cytokine; LPS-up."),
    ("cytokines_inflammatory", "Il27", "Inflammatory cytokine family member; LPS-up trend."),
    ("chemokines_migration", "Ccl3", "LPS-up inflammatory chemokine."),
    ("chemokines_migration", "Ccl4", "LPS-up inflammatory chemokine."),
    ("chemokines_migration", "Cxcl9", "IFNg-associated CXCR3 chemokine."),
    ("ifn_response", "Gbp4", "Strong IFNg-up interferon-stimulated GTPase."),
    ("ifn_response", "Isg15", "Canonical ISG retained as pathway anchor."),
    ("ifn_response", "Stat1", "Core IFN signaling transcription factor."),
    ("prr_nfkb", "Cd14", "LPS/TLR co-receptor; LPS-up PRR marker."),
    ("prr_nfkb", "Myd88", "TLR adaptor; primary PRR/NF-kB pathway gene."),
    ("prr_nfkb", "Nfkbia", "NF-kB feedback regulator."),
    ("stress_myeloid", "Lcn2", "LPS-up myeloid stress/inflammatory marker."),
    ("stress_myeloid", "S100a8", "LPS-up inflammatory myeloid stress marker."),
    ("stress_myeloid", "Lyz2", "Myeloid context/stress marker with treatment shifts."),
    ("dc_context", "Itgax", "CD11c/DC context marker; treatment-responsive in current DGE."),
    ("dc_context", "Batf3", "cDC1/DC lineage-context transcription factor."),
    ("dc_context", "Sirpa", "DC subset/context marker."),
]

T_CELL_SIGNATURES = {
    "TCR_core": ["Cd2", "Cd3d", "Cd3e", "Cd3g", "Cd247", "Trac", "Trbc1", "Trbc2", "Cd4", "Cd8a", "Cd8b1", "Lck", "Fyn", "Zap70", "Lat", "Lcp2", "Themis", "Ptpn6", "Ptpn22"],
    "Naive_memory": ["Sell", "Ccr7", "Tcf7", "Lef1", "Il7r", "Bcl2", "Klf2", "S1pr1", "S1pr5", "Satb1", "Ltb", "Cd27", "Cd28", "Cd44", "Ly6a", "Ly6c2", "Cxcr5", "Bach2"],
    "Activation_costim": ["Cd69", "Il2ra", "Il2rb", "Icos", "Cd40lg", "Tnfrsf4", "Tnfrsf9", "Tnfrsf18", "Tnfrsf25", "Cd226", "Nfkbia", "Nfkbiz", "Nfkbid", "Rel", "Rela", "Nfkb1", "Nfkb2", "Jun", "Junb", "Jund", "Fos", "Fosb", "Atf3", "Dusp1", "Dusp2", "Dusp4", "Dusp5"],
    "Cytotoxic_effector": ["Gzma", "Gzmb", "Gzmc", "Gzmk", "Gzmm", "Prf1", "Nkg7", "Ccl3", "Ccl4", "Ccl5", "Ifng", "Tnf", "Fasl", "Ctsw", "Klrg1", "Cx3cr1", "Zeb2", "Id2", "Slamf7", "Fcrl6"],
    "Inhibitory_exhaustion": ["Pdcd1", "Ctla4", "Lag3", "Havcr2", "Tigit", "Entpd1", "Cd160", "Btla", "Tox", "Tox2", "Nr4a1", "Nr4a2", "Nr4a3", "Batf", "Prdm1", "Eomes", "Slamf6", "Lair1", "Il10", "Ahr"],
    "Cytokine_signaling": ["Il2", "Il4", "Il10", "Il17a", "Il21", "Csf2", "Ifngr1", "Ifngr2", "Il12rb1", "Il12rb2", "Il18r1", "Il18rap", "Stat1", "Stat3", "Stat4", "Stat5a", "Stat5b", "Stat6", "Socs1", "Socs2", "Socs3", "Cish"],
    "Trafficking_adhesion": ["Cxcr3", "Cxcr4", "Cxcr6", "Ccr2", "Ccr5", "Ccr6", "Ccr9", "Itga1", "Itga4", "Itgae", "Itgal", "Itgam", "Itgb1", "Itgb2", "Itgb7", "Selplg", "Cd6", "Cd38", "Cd52"],
    "Proliferation": ["Mki67", "Top2a", "Stmn1", "Pcna", "Mcm2", "Mcm3", "Mcm4", "Mcm5", "Mcm6", "Mcm7", "Cdk1", "Cdk2", "Ccna2", "Ccnb1", "Ccnb2", "Cenpf", "Birc5", "Ube2c", "Hmgb2", "Tuba1b"],
    "TF_differentiation": ["Tbx21", "Eomes", "Prdm1", "Bcl6", "Rorc", "Gata3", "Foxp3", "Runx1", "Runx2", "Runx3", "Irf1", "Irf4", "Irf7", "Irf8", "Myc", "Hif1a", "Nfatc1", "Nfatc2", "Id2", "Id3", "Ikzf2", "Ikzf3", "Zbtb7b"],
    "IFN_stress": ["Isg15", "Ifit1", "Ifit2", "Ifit3", "Ifitm1", "Ifitm2", "Ifitm3", "Oas1a", "Oas1g", "Oas2", "Oas3", "Mx1", "Mx2", "Rsad2", "Rigi", "Ifih1", "Usp18", "Gbp2", "Gbp3", "Gbp4", "Gbp5", "Xaf1", "Irf9", "Hspa1a", "Hspa1b", "Hsp90aa1", "Hsp90ab1"],
    "Survival_metabolism": ["Bcl2l11", "Bcl2l1", "Mcl1", "Birc2", "Birc3", "Fas", "Casp3", "Casp8", "Tnfaip3", "Pim1", "Pim2", "Slc2a1", "Slc7a5", "Ldha", "Gapdh", "Hk2", "Pgk1", "Pkm", "Mtor", "Rptor", "Rictor", "Tsc1", "Tsc2"],
}

HIGHLIGHTED_GENE_GROUPS = {
    "Proliferation": ["Stmn1", "Pcna", "Birc5", "Mcm2", "Mcm3", "Mcm5", "Mcm6", "Cdk1", "Ube2c"],
    "Naive_recirc": ["Sell", "Klf2", "S1pr1", "Il7r", "Tcf7", "Lef1"],
    "IFN_stress": ["Ifit3", "Gbp2", "Gbp4", "Stat1", "Isg15", "Rigi", "Usp18"],
    "Cytotoxic_chemokine": ["Ccl3", "Ccl4", "Ifng", "Gzmb", "Nkg7", "Xcl1"],
    "Multi_contact_metabolic": ["Fabp5", "Mif", "Hspe1", "Srm"],
}
HIGHLIGHTED_GENES = [gene for genes in HIGHLIGHTED_GENE_GROUPS.values() for gene in genes]

GENERATED_TABLES = {
    "bubble_condition_global_unique_dcbc.csv",
    "bubble_dc_supported_denominator_by_condition.csv",
    "bubble_dc_supported_normalized_qualifying_cell_conditions.csv",
    "bubble_dc_supported_normalized_source.csv",
    "bubble_peptide_treatment_qualifying_cell_conditions.csv",
    "bubble_peptide_treatment_source.csv",
    "cell_dcbc_edge_table.csv",
    "dc_dge_cells_used.csv",
    "dc_dge_primary_gene_contrast_metrics.csv",
    "dc_dge_primary_gene_selection.csv",
    "dc_dge_program_summary.csv",
    "dc_dge_top_gene_heatmap_primary_source.csv",
    "dc_dge_top_gene_heatmap_source.csv",
    "dc_dge_volcano_IFNg_vs_no_treatment_source.csv",
    "dc_dge_volcano_LPS_vs_no_treatment_source.csv",
    "dc_dge_welch_results.csv.gz",
    "dc_phenotype_heatmap_mean_expression.csv",
    "dc_phenotype_heatmap_zscore.csv",
    "dc_supported_denominator_by_peptide_treatment_for_normalized_box.csv",
    "dc_treatment_assignment_90pct_purity.csv",
    "dc_treatment_assignment_90pct_purity_summary.csv",
    "dc_umi_per_cell_violin_source.csv",
    "figure_manifest.csv",
    "interaction_strength_distribution_curve_source.csv",
    "interaction_strength_distribution_source.csv",
    "peptide_umi_normalized_vs_bulk_lfc_source.csv",
    "single_cell_interaction_fingerprints_source.csv",
    "t_cell_dcbc_pickup_metrics.csv",
    "t_cell_peptide_heatmap_mean_expression.csv",
    "t_cell_peptide_heatmap_zscore.csv",
    "t_cell_select_gene_set_bubble_source.csv",
    "t_cell_signature_cell_scores.csv.gz",
    "t_cell_signature_gene_sets.csv",
    "t_cell_signature_group_median_heatmap_zscore.csv",
    "t_cell_signature_group_summary.csv",
    "t_cell_signature_pairwise_vs_no_interaction.csv",
    "t_cell_signature_peptide_order_single_cell_lfc.csv",
    "t_cell_treatment_heatmap_mean_expression.csv",
    "t_cell_treatment_heatmap_positive_mean_expression.csv",
    "t_cell_treatment_heatmap_zscore.csv",
    "t_cell_umi_per_cell_violin_source.csv",
    "tcell_treatment_dc_supported_normalized_peptide_collapsed_single_cell_source.csv",
    "tcell_treatment_dc_supported_normalized_peptide_condition_source.csv",
    "tcell_treatment_dc_supported_normalized_stats_vs_no_treatment.csv",
    "tcell_highlighted_gene_relative_pattern_bubble_source.csv",
    "tcell_peptide_dge_cells_used.csv",
    "tcell_peptide_dge_highlighted_gene_bubble_source.csv",
    "tcell_peptide_dge_summary_by_comparison.csv",
    "tcell_peptide_dge_vs_no_interaction.csv.gz",
    "tcell_signature_effect_ci_peptide_transfer_color_metric.csv",
    "tcell_signature_effect_ci_peptide_transfer_qualifying_cells.csv",
    "tcell_signature_effect_ci_vs_no_interaction_source.csv",
    "umap_single_cell_interaction_background_plot_points.csv",
    "umap_single_cell_interaction_links.csv",
    "umap_standard_plot_points.csv",
    "umap_standard_source.csv",
    "unique_dcbc_pickup_multiplicity_source.csv",
    "unique_dcbc_pickup_single_cell_dot_source.csv",
}

GENERATED_FIGURES = {
    "bubble_peptide_treatment_C57BL6",
    "bubble_peptide_treatment_OTI",
    "bubble_peptide_treatment_dc_supported_normalized_C57BL6",
    "bubble_peptide_treatment_dc_supported_normalized_OTI",
    "dc_dge_program_summary",
    "dc_dge_top_gene_heatmap_primary",
    "dc_dge_top_gene_heatmap",
    "dc_dge_volcano_IFNg_vs_no_treatment",
    "dc_dge_volcano_LPS_vs_no_treatment",
    "dc_phenotype_heatmap_clustered",
    "dc_total_barcode_umi_by_treatment_violin",
    "dc_umi_per_cell_by_treatment_violin",
    "gex_umap_retained_classified_cells",
    "interaction_fingerprint_C57BL6",
    "interaction_fingerprint_OTI",
    "interaction_strength_distribution",
    "peptide_umi_normalized_vs_bulk_lfc",
    "t_cell_peptide_heatmap_clustered",
    "t_cell_peptide_specific_umi_violin",
    "t_cell_select_gene_set_bubble",
    "t_cell_signature_group_median_heatmap",
    "t_cell_total_barcode_umi_by_type_violin",
    "t_cell_treatment_heatmap_clustered",
    "t_cell_umi_per_cell_violin",
    "tcell_dcbc_geomean_donor_normalized_bubble",
    "tcell_highlighted_gene_relative_pattern_bubble",
    "tcell_signature_effect_ci_vs_no_interaction",
    "tcell_treatment_dc_supported_normalized_peptide_collapsed_box_C57BL6",
    "tcell_treatment_dc_supported_normalized_peptide_collapsed_box_OTI",
    "top20_tcr_clonotype_counts",
    "umap_single_cell_interaction_map",
    "umap_standard_cell_types",
    "unique_dcbc_pickup_multiplicity_bar",
    "unique_dcbc_pickup_single_cell_dot",
}


def ensure_dirs() -> None:
    for path in [TABLES, QC, FIG_PDF, FIG_PNG]:
        path.mkdir(parents=True, exist_ok=True)
    for fig_name in GENERATED_FIGURES:
        for path, suffix in [(FIG_PDF, ".pdf"), (FIG_PNG, ".png")]:
            old_file = path / f"{fig_name}{suffix}"
            if old_file.exists():
                old_file.unlink()
    for table_name in GENERATED_TABLES:
        old_table = TABLES / table_name
        if old_table.exists():
            old_table.unlink()


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def save_dual(fig: plt.Figure, name: str, pdf_dpi: int = 300, png_dpi: int = 300) -> dict[str, str]:
    pdf = FIG_PDF / f"{name}.pdf"
    png = FIG_PNG / f"{name}.png"
    fig.savefig(pdf, dpi=pdf_dpi, bbox_inches="tight")
    fig.savefig(png, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def load_config() -> dict[str, Any]:
    return json.loads((PUB / "code" / "config.yaml").read_text())


def resolve_pub_path(maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (PUB / path).resolve()


def read_analysis_csv(analysis_tar: Path, member_name: str) -> pd.DataFrame:
    with tarfile.open(analysis_tar, "r:gz") as tar:
        fh = tar.extractfile(member_name)
        if fh is None:
            raise FileNotFoundError(member_name)
        return pd.read_csv(fh)


def build_umap_source(config: dict[str, Any], cell_meta: pd.DataFrame, t_meta: pd.DataFrame) -> pd.DataFrame:
    analysis_path = resolve_pub_path(config["paths"]["cellranger_analysis"])
    umap = read_analysis_csv(analysis_path, "umap/gene_expression_2_components/projection.csv")
    retained = cell_meta[cell_meta["cell_class"].isin(["dendritic_cell", "t_cell"])].copy()
    t_type = dict(zip(t_meta["CellBC"].astype(str), t_meta["t_cell_type"].astype(str)))
    labels = []
    for row in retained.itertuples(index=False):
        if row.cell_class == "dendritic_cell":
            labels.append("dendritic cell")
        else:
            labels.append("OTI T cell" if t_type.get(row.CellBC) == "OTI" else "C57BL/6 T cell")
    retained["plot_label"] = labels
    plot_df = retained.merge(umap, left_on="CellBC", right_on="Barcode", how="inner")
    plot_df.to_csv(TABLES / "figure_umap_retained_cells_source.csv", index=False)
    return plot_df


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    cell_meta = pd.read_csv(PUB / "data_intermediate" / "cell_metadata.csv", dtype={"CellBC": str})
    t_meta = pd.read_csv(PUB / "data_intermediate" / "t_cell_metadata.csv", dtype={"CellBC": str}).fillna("")
    dc_meta = pd.read_csv(PUB / "data_intermediate" / "dendritic_cell_metadata.csv", dtype={"CellBC": str}).fillna("")
    dcbc_identity = pd.read_csv(PUB / "data_intermediate" / "dcbc_identity_table.csv", dtype=str).fillna("")
    counts = pd.read_csv(
        PUB / "data_intermediate" / "barcode_read_support_filtered_count_table.csv.gz",
        dtype={"CellBC": str, "DCBC": str, "PeptideBC_Index": str, "PeptideBC_Name": str, "TreatmentBC": str},
    )
    umap = build_umap_source(config, cell_meta, t_meta)
    return {
        "cell_meta": cell_meta,
        "t_meta": t_meta,
        "dc_meta": dc_meta,
        "dcbc_identity": dcbc_identity,
        "counts": counts,
        "umap": umap,
    }


def build_edge_table(inputs: dict[str, Any]) -> pd.DataFrame:
    counts = inputs["counts"]
    cell_meta = inputs["cell_meta"][["CellBC", "cell_class"]].copy()
    identity_cols = [
        "DCBC",
        "dcbc_identity_status",
        "AssignedPeptideBC_Index",
        "AssignedPeptideBC_Name",
        "AssignedTreatmentBC",
        "AssignedTreatment",
    ]
    identity = inputs["dcbc_identity"][identity_cols].copy()
    edges = counts.groupby(["CellBC", "DCBC"], as_index=False)[["UMI", "Reads"]].sum()
    edges = edges.merge(cell_meta, on="CellBC", how="left").merge(identity, on="DCBC", how="left").fillna("")
    edges["assigned_dcbc"] = (
        (edges["dcbc_identity_status"] == "assigned")
        & edges["AssignedPeptideBC_Name"].ne("")
        & edges["AssignedTreatment"].ne("")
    )
    edges.to_csv(TABLES / "cell_dcbc_edge_table.csv", index=False)
    return edges


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    ok_idx = np.where(ok)[0]
    out[ok_idx[order]] = adj
    return out


def geometric_mean(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[arr > 0]
    return float(np.exp(np.mean(np.log(arr)))) if len(arr) else 0.0


def tcell_signature_effect_transfer_color_metric(
    edges: pd.DataFrame,
    t_meta: pd.DataFrame,
    group_order: list[str],
) -> pd.DataFrame:
    assigned_flag = edges["assigned_dcbc"].fillna(False)
    if assigned_flag.dtype != bool:
        assigned_flag = assigned_flag.astype(str).str.lower().isin({"true", "1", "yes"})
    assigned = edges[assigned_flag].copy()

    dc_edges = assigned[assigned["cell_class"].eq("dendritic_cell")].copy()
    denom = (
        dc_edges.groupby("AssignedPeptideBC_Name", as_index=False)
        .agg(
            dc_supported_unique_dcbc=("DCBC", "nunique"),
            dc_supported_total_dcbc_umi=("UMI", "sum"),
            dc_supported_total_dcbc_reads=("Reads", "sum"),
        )
        .rename(columns={"AssignedPeptideBC_Name": "peptide_group"})
    )

    t_edges = assigned[assigned["cell_class"].eq("t_cell")].copy()
    t_cell_types = t_meta[["CellBC", "t_cell_type"]].drop_duplicates("CellBC")
    t_edges = t_edges.merge(t_cell_types, on="CellBC", how="left")
    t_edges = t_edges[t_edges["t_cell_type"].eq("OTI")].copy()

    qualifying_cols = [
        "CellBC",
        "peptide_group",
        "peptide_umi",
        "peptide_reads",
        "peptide_unique_dcbc",
        "total_assigned_dcbc_umi",
        "total_assigned_dcbc_reads",
        "total_assigned_unique_dcbc",
        "peptide_umi_fraction",
    ]
    if t_edges.empty:
        qualifying = pd.DataFrame(columns=qualifying_cols)
    else:
        cell_peptide = (
            t_edges.groupby(["CellBC", "AssignedPeptideBC_Name"], as_index=False)
            .agg(
                peptide_umi=("UMI", "sum"),
                peptide_reads=("Reads", "sum"),
                peptide_unique_dcbc=("DCBC", "nunique"),
            )
            .rename(columns={"AssignedPeptideBC_Name": "peptide_group"})
        )
        cell_total = (
            t_edges.groupby("CellBC", as_index=False)
            .agg(
                total_assigned_dcbc_umi=("UMI", "sum"),
                total_assigned_dcbc_reads=("Reads", "sum"),
                total_assigned_unique_dcbc=("DCBC", "nunique"),
            )
        )
        cell_peptide = cell_peptide.merge(cell_total, on="CellBC", how="left")
        cell_peptide["peptide_umi_fraction"] = (
            cell_peptide["peptide_umi"].to_numpy(dtype=float)
            / np.maximum(cell_peptide["total_assigned_dcbc_umi"].to_numpy(dtype=float), 1.0)
        )
        qualifying = cell_peptide[
            (cell_peptide["peptide_umi"] > PEPTIDE_TRANSFER_UMI_MIN_EXCLUSIVE)
            & (cell_peptide["peptide_umi_fraction"] >= PEPTIDE_TRANSFER_FRACTION_MIN)
        ].copy()
        qualifying = qualifying.reindex(columns=qualifying_cols)
    qualifying.to_csv(TABLES / "tcell_signature_effect_ci_peptide_transfer_qualifying_cells.csv", index=False)

    rows = []
    for peptide in [group for group in group_order if group != "multi_peptide"]:
        vals = qualifying.loc[qualifying["peptide_group"].eq(peptide), "peptide_umi"].to_numpy(dtype=float)
        denom_row = denom[denom["peptide_group"].eq(peptide)]
        dc_total_umi = int(denom_row["dc_supported_total_dcbc_umi"].iloc[0]) if len(denom_row) else 0
        dc_total_reads = int(denom_row["dc_supported_total_dcbc_reads"].iloc[0]) if len(denom_row) else 0
        dc_unique = int(denom_row["dc_supported_unique_dcbc"].iloc[0]) if len(denom_row) else 0
        geomean = geometric_mean(vals)
        normalized = float(geomean / dc_total_umi * PEPTIDE_TRANSFER_SCALE) if dc_total_umi else np.nan
        rows.append(
            {
                "peptide_group": peptide,
                "qualifying_oti_cells": int(len(vals)),
                "geomean_dominant_peptide_umi": geomean,
                "dc_supported_unique_dcbc": dc_unique,
                "dc_supported_total_dcbc_umi": dc_total_umi,
                "dc_supported_total_dcbc_reads": dc_total_reads,
                "geomean_dominant_peptide_umi_per_10k_dc_dcbc_umi": normalized,
                "cell_peptide_umi_min_exclusive": PEPTIDE_TRANSFER_UMI_MIN_EXCLUSIVE,
                "cell_peptide_fraction_min": PEPTIDE_TRANSFER_FRACTION_MIN,
                "normalization_scale": PEPTIDE_TRANSFER_SCALE,
            }
        )
    transfer = pd.DataFrame(rows)
    transfer.to_csv(TABLES / "tcell_signature_effect_ci_peptide_transfer_color_metric.csv", index=False)
    return transfer


def zscore_series(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return values * 0
    return (values - values.mean()) / std


def dc_supported_condition_tables(
    edges: pd.DataFrame,
    t_meta: pd.DataFrame,
    peptide_order: list[str],
    treatment_order: list[str],
    min_condition_umi: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dc_edges = edges[
        (edges["cell_class"] == "dendritic_cell")
        & edges["assigned_dcbc"]
        & edges["AssignedTreatment"].isin(treatment_order)
    ].copy()
    supported_dcbc = set(dc_edges["DCBC"].astype(str))
    denom = (
        dc_edges.groupby(["AssignedPeptideBC_Name", "AssignedTreatment"], as_index=False)
        .agg(
            dc_supported_unique_dcbc=("DCBC", "nunique"),
            dc_supported_total_dcbc_umi=("UMI", "sum"),
            dc_supported_total_dcbc_reads=("Reads", "sum"),
        )
        .rename(columns={"AssignedPeptideBC_Name": "PeptideBC_Name", "AssignedTreatment": "treatment"})
    )
    denom_map = {(r.PeptideBC_Name, r.treatment): r for r in denom.itertuples(index=False)}
    denom_rows = []
    for peptide in peptide_order:
        for treatment in treatment_order:
            row = denom_map.get((peptide, treatment))
            denom_rows.append(
                {
                    "PeptideBC_Name": peptide,
                    "treatment": treatment,
                    "dc_supported_unique_dcbc": int(getattr(row, "dc_supported_unique_dcbc", 0)),
                    "dc_supported_total_dcbc_umi": int(getattr(row, "dc_supported_total_dcbc_umi", 0)),
                    "dc_supported_total_dcbc_reads": int(getattr(row, "dc_supported_total_dcbc_reads", 0)),
                }
            )
    denom_full = pd.DataFrame(denom_rows)

    t_edges = edges[
        (edges["cell_class"] == "t_cell")
        & edges["assigned_dcbc"]
        & edges["AssignedTreatment"].isin(treatment_order)
        & edges["DCBC"].astype(str).isin(supported_dcbc)
    ].copy()
    t_edges = t_edges.merge(t_meta[["CellBC", "t_cell_type"]], on="CellBC", how="left")
    condition = (
        t_edges.groupby(["CellBC", "t_cell_type", "AssignedPeptideBC_Name", "AssignedTreatment"], as_index=False)
        .agg(
            condition_umi=("UMI", "sum"),
            condition_reads=("Reads", "sum"),
            condition_unique_supported_dcbc=("DCBC", "nunique"),
        )
        .rename(columns={"AssignedPeptideBC_Name": "PeptideBC_Name", "AssignedTreatment": "treatment"})
    )
    condition = condition[condition["condition_umi"] >= min_condition_umi].copy()
    condition = condition.merge(denom_full, on=["PeptideBC_Name", "treatment"], how="left")
    condition["condition_umi_per_10k_dc_dcbc_umi"] = (
        condition["condition_umi"].to_numpy(dtype=float)
        / np.maximum(condition["dc_supported_total_dcbc_umi"].to_numpy(dtype=float), 1.0)
        * 10000.0
    )
    condition["cell_condition_min_umi"] = min_condition_umi
    return denom_full, condition, t_edges


def generate_dc_supported_normalized_bubble(
    config: dict[str, Any],
    inputs: dict[str, Any],
    edges: pd.DataFrame,
    peptide_order: list[str],
    treatment_order: list[str],
    manifest: list[dict[str, str]],
) -> dict[str, Any]:
    min_condition_umi = int(config.get("thresholds", {}).get("bubble_min_condition_umi", 2))
    denom, condition, t_edges = dc_supported_condition_tables(edges, inputs["t_meta"], peptide_order, treatment_order, min_condition_umi)
    denom_path = TABLES / "bubble_dc_supported_denominator_by_condition.csv"
    condition_path = TABLES / "bubble_dc_supported_normalized_qualifying_cell_conditions.csv"
    denom.to_csv(denom_path, index=False)
    condition.to_csv(condition_path, index=False)

    rows = []
    for t_type in T_CELL_TYPES:
        for peptide_i, peptide in enumerate(peptide_order):
            for treatment_i, treatment in enumerate(treatment_order):
                sub = condition[
                    (condition["t_cell_type"] == t_type)
                    & (condition["PeptideBC_Name"] == peptide)
                    & (condition["treatment"] == treatment)
                ]
                vals = sub["condition_umi"].to_numpy(dtype=float)
                denom_row = denom[(denom["PeptideBC_Name"] == peptide) & (denom["treatment"] == treatment)]
                donor_unique = int(denom_row["dc_supported_unique_dcbc"].iloc[0]) if len(denom_row) else 0
                donor_total_umi = int(denom_row["dc_supported_total_dcbc_umi"].iloc[0]) if len(denom_row) else 0
                geomean_umi = geometric_mean(vals)
                rows.append(
                    {
                        "t_cell_type": t_type,
                        "PeptideBC_Name": peptide,
                        "AssignedTreatment": treatment,
                        "peptide_order": peptide_i,
                        "treatment_order": treatment_i,
                        "qualifying_cells": int(len(vals)),
                        "dc_supported_unique_dcbc": donor_unique,
                        "dc_supported_total_dcbc_umi": donor_total_umi,
                        "qualifying_cells_per_dc_supported_unique_dcbc": float(len(vals) / donor_unique) if donor_unique else 0.0,
                        "geomean_supported_condition_umi": geomean_umi,
                        "geomean_condition_umi_per_10k_dc_dcbc_umi": float(geomean_umi / donor_total_umi * 10000.0) if donor_total_umi else 0.0,
                        "cell_condition_min_umi": min_condition_umi,
                    }
                )
    bubble = pd.DataFrame(rows)
    source = TABLES / "bubble_dc_supported_normalized_source.csv"
    bubble.to_csv(source, index=False)

    color_col = "geomean_condition_umi_per_10k_dc_dcbc_umi"
    size_col = "qualifying_cells_per_dc_supported_unique_dcbc"
    positive = bubble.loc[bubble[color_col] > 0, color_col].to_numpy(float)
    vmin = float(np.percentile(positive, 5)) if len(positive) else 0.0
    vmax = float(np.percentile(positive, 90)) if len(positive) else 1.0
    vmax = vmax if vmax > vmin else vmin + 1e-9
    size_max = max(float(bubble[size_col].max()), 1e-12)
    size_values = bubble.loc[bubble[size_col] > 0, size_col].to_numpy(float)
    size_legend_values = np.unique(np.round(np.percentile(size_values, [25, 50, 90]), 2)) if len(size_values) else np.array([])
    cmap = LinearSegmentedColormap.from_list("dc_supported_transfer_red", ["#f7fbff", "#f9d6d6", "#ed8590", "#c83f58"])

    for t_type in T_CELL_TYPES:
        sub = bubble[bubble["t_cell_type"] == t_type].copy()
        plot_sub = sub[sub[color_col] > 0].copy()
        fig, ax = plt.subplots(figsize=(2.9, 4.6))
        ax.set_axisbelow(True)
        if len(plot_sub):
            sizes = 18 + (plot_sub[size_col].to_numpy(float) / size_max) * 230
            colors = np.clip(plot_sub[color_col].to_numpy(float), vmin, vmax)
            ax.scatter(plot_sub["treatment_order"], plot_sub["peptide_order"], s=sizes, c=colors, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="#7a2f36", linewidths=0.35, zorder=3)
        ax.set_xticks(range(len(treatment_order)))
        ax.set_xticklabels(treatment_order, rotation=35, ha="right")
        ax.set_yticks(range(len(peptide_order)))
        ax.set_yticklabels(peptide_order)
        ax.set_ylim(len(peptide_order) - 0.5, -0.5)
        ax.set_xlim(-0.5, len(treatment_order) - 0.5)
        ax.grid(True, color="#e8e8e8", linewidth=0.35, zorder=0)
        ax.set_title(f"{t_type} DC-supported")
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Peptide")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.055, pad=0.04)
        cbar.set_label("Geomean UMI / 10k DC UMI", fontsize=6)
        if len(size_legend_values):
            handles = [
                ax.scatter([], [], s=18 + (float(value) / size_max) * 230, facecolor="#f9d6d6", edgecolor="#7a2f36", linewidths=0.35)
                for value in size_legend_values
            ]
            ax.legend(
                handles,
                [f"{value:g}" for value in size_legend_values],
                title="Cells / DC-supported DCBC",
                fontsize=5,
                title_fontsize=5,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=len(size_legend_values),
                columnspacing=1.0,
                handletextpad=0.7,
            )
        files = save_dual(fig, f"bubble_peptide_treatment_dc_supported_normalized_{t_type}")
        manifest.append({"figure": f"DC-supported normalized bubble {t_type}", "source_table": str(source), **files, "script_function": "generate_dc_supported_normalized_bubble"})

    summary = {
        "source_count_table": str(PUB / "data_intermediate" / "barcode_read_support_filtered_count_table.csv.gz"),
        "source_table": str(source),
        "denominator_table": str(denom_path),
        "qualifying_cell_condition_table": str(condition_path),
        "normalization_rule": "Color metric is geometric mean T-cell condition UMI divided by dendritic-cell total DCBC UMI for the same peptide+treatment, reported per 10k DC DCBC UMI.",
        "cell_condition_min_umi": min_condition_umi,
        "n_bubble_rows": int(len(bubble)),
        "n_nonzero_bubbles": int((bubble[color_col] > 0).sum()),
        "color_min": vmin,
        "color_max": vmax,
        "t_cell_dcbc_edges_after_dc_support_filter": int(len(t_edges)),
    }
    write_json(summary, QC / "bubble_dc_supported_normalized_summary.json")
    return {"bubble_dc_supported_normalized": summary}


def generate_tcell_dc_supported_normalized_boxplots(
    config: dict[str, Any],
    inputs: dict[str, Any],
    edges: pd.DataFrame,
    peptide_order: list[str],
    treatment_order: list[str],
    manifest: list[dict[str, str]],
) -> dict[str, Any]:
    min_condition_umi = int(config.get("thresholds", {}).get("bubble_min_condition_umi", 2))
    denom, condition, _t_edges = dc_supported_condition_tables(edges, inputs["t_meta"], peptide_order, treatment_order, min_condition_umi)
    denom_path = TABLES / "dc_supported_denominator_by_peptide_treatment_for_normalized_box.csv"
    condition_path = TABLES / "tcell_treatment_dc_supported_normalized_peptide_condition_source.csv"
    denom.to_csv(denom_path, index=False)
    condition.to_csv(condition_path, index=False)

    metric = "treatment_umi_per_10k_dc_dcbc_umi"
    collapsed = (
        condition.groupby(["CellBC", "t_cell_type", "treatment"], as_index=False)
        .agg(
            treatment_umi_per_10k_dc_dcbc_umi=("condition_umi_per_10k_dc_dcbc_umi", "sum"),
            treatment_raw_umi_from_qualifying_peptides=("condition_umi", "sum"),
            n_qualifying_peptides=("PeptideBC_Name", "nunique"),
        )
    )
    source = TABLES / "tcell_treatment_dc_supported_normalized_peptide_collapsed_single_cell_source.csv"
    collapsed.to_csv(source, index=False)

    stat_rows = []
    for t_type in T_CELL_TYPES:
        sub = collapsed[collapsed["t_cell_type"] == t_type]
        ref = sub.loc[sub["treatment"] == "no_treatment", metric].to_numpy(float)
        p_values = []
        row_start = len(stat_rows)
        for treatment in ["LPS", "PolyIC", "IFNg"]:
            vals = sub.loc[sub["treatment"] == treatment, metric].to_numpy(float)
            if len(vals) and len(ref):
                test = stats.mannwhitneyu(vals, ref, alternative="two-sided", method="asymptotic")
                p = float(test.pvalue)
                common = float(test.statistic / (len(vals) * len(ref)))
            else:
                p = math.nan
                common = math.nan
            p_values.append(p)
            stat_rows.append(
                {
                    "t_cell_type": t_type,
                    "treatment": treatment,
                    "reference": "no_treatment",
                    "n_treatment": int(len(vals)),
                    "n_reference": int(len(ref)),
                    "median_treatment_umi_per_10k_dc_dcbc_umi": float(np.median(vals)) if len(vals) else math.nan,
                    "median_reference_umi_per_10k_dc_dcbc_umi": float(np.median(ref)) if len(ref) else math.nan,
                    "geomean_treatment_umi_per_10k_dc_dcbc_umi": geometric_mean(vals),
                    "geomean_reference_umi_per_10k_dc_dcbc_umi": geometric_mean(ref),
                    "common_language_effect_treatment_gt_reference": common,
                    "p_value": p,
                }
            )
        q = bh_adjust(np.asarray(p_values, dtype=float))
        for i, q_value in enumerate(q):
            stat_rows[row_start + i]["p_adj_bh_by_t_cell_type"] = float(q_value)
            stat_rows[row_start + i]["q_label"] = "q<0.001" if np.isfinite(q_value) and q_value < 0.001 else f"q={q_value:.3f}"
    stat_table = pd.DataFrame(stat_rows)
    stats_path = TABLES / "tcell_treatment_dc_supported_normalized_stats_vs_no_treatment.csv"
    stat_table.to_csv(stats_path, index=False)

    for t_type in T_CELL_TYPES:
        sub = collapsed[collapsed["t_cell_type"] == t_type]
        data = [sub.loc[sub["treatment"] == trt, metric].to_numpy(float) for trt in treatment_order]
        fig, ax = plt.subplots(figsize=(3.45, 2.6))
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.45)
        for patch, trt in zip(bp["boxes"], treatment_order):
            patch.set(facecolor=TREATMENT_COLORS.get(trt, "#cfcfcf"), edgecolor="#3f3f3f", alpha=0.85, linewidth=0.7)
        for element in ["whiskers", "caps", "medians"]:
            for artist in bp[element]:
                artist.set(color="#3f3f3f", linewidth=0.7)
        for i, vals in enumerate(data, start=1):
            if len(vals):
                ax.scatter(i, geometric_mean(vals), marker="D", s=9, color="#7a2f36", edgecolor="#3f3f3f", linewidth=0.25, zorder=4)
            ax.text(i, -0.16, f"n={len(vals):,}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(treatment_order) + 1))
        ax.set_xticklabels(["no_treat", "LPS", "PolyIC", "IFNg"], rotation=0)
        ax.set_ylabel("T-cell DCBC UMI / 10k DC UMI")
        ax.set_title(f"{t_type} peptide-collapsed, DC-supported")
        stat_sub = stat_table[stat_table["t_cell_type"] == t_type]
        y_top = ax.get_ylim()[1]
        for i, treatment in enumerate(["LPS", "PolyIC", "IFNg"], start=2):
            labels = stat_sub.loc[stat_sub["treatment"] == treatment, "q_label"]
            label = labels.iloc[0] if len(labels) else "q=NA"
            ax.text(i, y_top / 1.25, label, ha="center", va="center", fontsize=6)
        files = save_dual(fig, f"tcell_treatment_dc_supported_normalized_peptide_collapsed_box_{t_type}")
        manifest.append({"figure": f"T cell DC-supported normalized treatment boxplot {t_type}", "source_table": str(source), **files, "stats_table": str(stats_path), "script_function": "generate_tcell_dc_supported_normalized_boxplots"})

    summary = {
        "source_count_table": str(PUB / "data_intermediate" / "barcode_read_support_filtered_count_table.csv.gz"),
        "peptide_condition_source": str(condition_path),
        "peptide_collapsed_source": str(source),
        "stats_table": str(stats_path),
        "normalization_rule": "Normalize each CellBC+peptide+treatment condition by DC-supported total UMI for the same peptide+treatment and multiply by 10,000; then sum normalized values across peptides within CellBC+treatment.",
        "cell_condition_min_umi": min_condition_umi,
        "n_collapsed_cell_treatment_rows": int(len(collapsed)),
    }
    write_json(summary, QC / "tcell_treatment_dc_supported_normalized_box_summary.json")
    return {"tcell_treatment_dc_supported_normalized_boxplots": summary}


def assign_dc_treatments_by_purity(edges: pd.DataFrame, cell_meta: pd.DataFrame, purity_threshold: float) -> pd.DataFrame:
    assigned_flag = edges["assigned_dcbc"].fillna(False)
    if assigned_flag.dtype != bool:
        assigned_flag = assigned_flag.astype(str).str.lower().isin({"true", "1", "yes"})
    dc_edges = edges[(edges["cell_class"] == "dendritic_cell") & assigned_flag].copy()
    support = dc_edges.groupby(["CellBC", "AssignedTreatment"], as_index=False).agg(treatment_umi=("UMI", "sum"), treatment_reads=("Reads", "sum"), unique_dcbc=("DCBC", "nunique"))
    dc_cells = cell_meta[cell_meta["cell_class"] == "dendritic_cell"].copy()
    assignment = dc_cells[["CellBC"] + [c for c in ["n_counts", "n_genes"] if c in dc_cells.columns]].copy()
    for treatment in TREATMENT_ORDER:
        tmp = support[support["AssignedTreatment"] == treatment][["CellBC", "treatment_umi"]].rename(columns={"treatment_umi": f"umi_{treatment}"})
        assignment = assignment.merge(tmp, on="CellBC", how="left")
        assignment[f"umi_{treatment}"] = assignment[f"umi_{treatment}"].fillna(0).astype(int)
    umi_cols = [f"umi_{t}" for t in TREATMENT_ORDER]
    assignment["total_assigned_treatment_umi"] = assignment[umi_cols].sum(axis=1)
    umi_values = assignment[umi_cols].to_numpy(dtype=float)
    top_idx = np.argmax(umi_values, axis=1) if len(assignment) else np.array([], dtype=int)
    top_umi = umi_values[np.arange(len(assignment)), top_idx] if len(assignment) else np.array([], dtype=float)
    total_umi = assignment["total_assigned_treatment_umi"].to_numpy(dtype=float)
    assignment["top_treatment"] = [TREATMENT_ORDER[int(i)] if total > 0 else "" for i, total in zip(top_idx, total_umi)]
    assignment["top_treatment_umi"] = top_umi.astype(int)
    assignment["top_treatment_fraction"] = np.divide(top_umi, total_umi, out=np.zeros_like(top_umi, dtype=float), where=total_umi > 0)
    assignment["n_treatments_with_umi"] = assignment[umi_cols].gt(0).sum(axis=1).astype(int)
    assignment["treatment_group"] = np.where(
        assignment["top_treatment_fraction"].ge(purity_threshold) & assignment["total_assigned_treatment_umi"].gt(0),
        assignment["top_treatment"],
        "",
    )
    assignment["assignment_status"] = np.where(
        assignment["total_assigned_treatment_umi"].eq(0),
        "no_assigned_treatment_umi",
        np.where(assignment["top_treatment_fraction"].ge(purity_threshold), "single_purity_treatment", "below_purity"),
    )
    assignment["dc_dge_treatment_purity_threshold"] = float(purity_threshold)
    return assignment


def unique_gene_expression(features: pd.DataFrame, matrix: sparse.csc_matrix) -> tuple[pd.DataFrame, sparse.csc_matrix]:
    gene_mask = features["feature_type"].fillna("Gene Expression").eq("Gene Expression").to_numpy()
    features = features.loc[gene_mask].copy().reset_index(drop=True)
    matrix = matrix[gene_mask, :].tocsc()
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    features["total_counts_selected"] = totals
    features["row_index"] = np.arange(len(features))
    keep = (
        features.sort_values(["gene_name", "total_counts_selected"], ascending=[True, False])
        .drop_duplicates("gene_name")["row_index"]
        .to_numpy()
    )
    keep = np.sort(keep)
    return features.iloc[keep].reset_index(drop=True), matrix[keep, :].tocsc()


def sparse_row_mean_var(matrix: sparse.csc_matrix, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = matrix[:, idx].tocsc()
    mean = np.asarray(sub.mean(axis=1)).ravel()
    sq = sub.copy()
    sq.data = sq.data**2
    mean_sq = np.asarray(sq.mean(axis=1)).ravel()
    n = max(len(idx), 1)
    var = np.maximum((mean_sq - mean**2) * n / max(n - 1, 1), 0)
    return mean, var


def plot_dc_dge_top_gene_heatmap_primary(dge: pd.DataFrame, contrasts: list[str], manifest: list[dict[str, str]]) -> dict[str, Any]:
    selected = pd.DataFrame(PRIMARY_DC_DGE_GENES, columns=["program", "gene", "selection_rationale"])
    selected["program_order"] = pd.Categorical(
        selected["program"],
        categories=list(dict.fromkeys(selected["program"].tolist())),
        ordered=True,
    ).codes
    selected["gene_order"] = np.arange(len(selected))

    dge_sel = dge[dge["gene"].isin(selected["gene"])].copy()
    metric_rows = []
    for gene in selected["gene"]:
        gene_rows = dge_sel[dge_sel["gene"] == gene]
        control_rows = gene_rows[gene_rows["contrast"].isin([f"{t}_vs_no_treatment" for t in contrasts])]
        if len(control_rows):
            metric_rows.append(
                {
                    "gene": gene,
                    "treatment_group": "no_treatment",
                    "mean_log1p": float(control_rows["mean_log1p_control"].iloc[0]),
                }
            )
        for treatment in contrasts:
            row = gene_rows[gene_rows["contrast"] == f"{treatment}_vs_no_treatment"]
            if len(row):
                metric_rows.append(
                    {
                        "gene": gene,
                        "treatment_group": treatment,
                        "mean_log1p": float(row["mean_log1p_treatment"].iloc[0]),
                    }
                )

    heatmap = pd.DataFrame(metric_rows).drop_duplicates(["gene", "treatment_group"], keep="first")
    heatmap = heatmap.merge(selected, on="gene", how="left")
    heatmap["z"] = heatmap.groupby("gene")["mean_log1p"].transform(zscore_series)

    contrast_metrics = []
    for treatment in contrasts:
        sub = dge_sel[dge_sel["contrast"] == f"{treatment}_vs_no_treatment"].copy()
        if sub.empty:
            continue
        sub = sub.merge(selected[["program", "gene", "gene_order"]], on="gene", how="left")
        contrast_metrics.append(
            sub[
                [
                    "program",
                    "gene",
                    "gene_order",
                    "contrast",
                    "tested",
                    "log2fc_mean_norm",
                    "padj_bh",
                    "pct_treatment",
                    "pct_control",
                    "mean_log1p_treatment",
                    "mean_log1p_control",
                ]
            ]
        )
    metrics = pd.concat(contrast_metrics, ignore_index=True) if contrast_metrics else pd.DataFrame()

    columns = ["no_treatment"] + list(contrasts)
    pivot = heatmap.pivot(index="gene", columns="treatment_group", values="z").reindex(
        index=selected["gene"].tolist(),
        columns=columns,
    )
    z_matrix = pivot.fillna(0).to_numpy(dtype=float)
    if len(pivot) > 1:
        linkage = hierarchy.linkage(z_matrix, method="average", metric="euclidean", optimal_ordering=True)
        row_order = hierarchy.leaves_list(linkage)
        cluster_method = "average_linkage_euclidean_on_gene_z_scores"
    else:
        row_order = np.arange(len(pivot))
        cluster_method = "not_clustered_single_gene"
    pivot = pivot.iloc[row_order]

    cluster_order = {gene: int(i) for i, gene in enumerate(pivot.index)}
    selected["zscore_cluster_order"] = selected["gene"].map(cluster_order)
    heatmap["zscore_cluster_order"] = heatmap["gene"].map(cluster_order)
    heatmap = heatmap.sort_values(["zscore_cluster_order", "treatment_group"]).reset_index(drop=True)
    if not metrics.empty:
        metrics["zscore_cluster_order"] = metrics["gene"].map(cluster_order)
        metrics = metrics.sort_values(["zscore_cluster_order", "contrast"])

    selection_path = TABLES / "dc_dge_primary_gene_selection.csv"
    heatmap_path = TABLES / "dc_dge_top_gene_heatmap_primary_source.csv"
    metrics_path = TABLES / "dc_dge_primary_gene_contrast_metrics.csv"
    selected.to_csv(selection_path, index=False)
    heatmap.to_csv(heatmap_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    fig_h = max(2.9, 0.135 * len(pivot))
    fig, ax = plt.subplots(figsize=(2.1, fig_h))
    cmap = LinearSegmentedColormap.from_list("primary_dc_dge_heatmap", [HEATMAP_LOW, HEATMAP_MID, HEATMAP_HIGH])
    im = ax.imshow(np.clip(pivot.to_numpy(float), -2.5, 2.5), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=5.8)
    ax.set_title("Primary DC program genes", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="z")
    files = save_dual(fig, "dc_dge_top_gene_heatmap_primary")
    manifest.append({"figure": "DC DGE primary gene heatmap", "source_table": str(heatmap_path), **files, "script_function": "run_dc_treatment_dge"})

    summary = {
        "n_primary_genes": int(selected["gene"].nunique()),
        "n_programs": int(selected["program"].nunique()),
        "treatment_groups": columns,
        "row_clustering": cluster_method,
        "selection_rule": "Curated primary genes from DC program classes; selected for biological interpretability and current DGE relevance.",
        "outputs": {
            "selection_table": str(selection_path),
            "heatmap_source": str(heatmap_path),
            "contrast_metrics": str(metrics_path),
            **files,
        },
    }
    write_json(summary, QC / "dc_dge_top_gene_heatmap_primary_summary.json")
    return summary


def run_dc_treatment_dge(config: dict[str, Any], inputs: dict[str, Any], edges: pd.DataFrame, manifest: list[dict[str, str]]) -> dict[str, Any]:
    thresholds = config.get("thresholds", {})
    purity_threshold = float(thresholds.get("dc_dge_treatment_purity_threshold", 0.90))
    min_pct = float(thresholds.get("dc_dge_min_pct", 0.05))
    min_total_counts = int(thresholds.get("dc_dge_min_total_counts", 10))
    contrasts = list(thresholds.get("dc_dge_contrasts", ["LPS", "IFNg"]))
    pseudocount = float(thresholds.get("dc_dge_pseudocount", 0.1))

    assignment = assign_dc_treatments_by_purity(edges, inputs["cell_meta"], purity_threshold)
    assignment_path = TABLES / "dc_treatment_assignment_90pct_purity.csv"
    assignment_summary_path = TABLES / "dc_treatment_assignment_90pct_purity_summary.csv"
    assignment.to_csv(assignment_path, index=False)
    assignment.groupby(["assignment_status", "treatment_group"], dropna=False).size().reset_index(name="n_cells").to_csv(assignment_summary_path, index=False)

    features, barcodes, matrix = load_10x_matrix(config)
    dge_cells = assignment[
        (assignment["assignment_status"] == "single_purity_treatment")
        & assignment["treatment_group"].isin(["no_treatment"] + contrasts)
    ].copy()
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    dge_cells["matrix_col"] = dge_cells["CellBC"].map(barcode_to_idx)
    dge_cells = dge_cells.dropna(subset=["matrix_col"]).copy()
    dge_cells["matrix_col"] = dge_cells["matrix_col"].astype(int)
    dge_cells = dge_cells.sort_values(["treatment_group", "CellBC"]).reset_index(drop=True)

    expr = matrix[:, dge_cells["matrix_col"].to_numpy()].tocsc()
    features, expr = unique_gene_expression(features, expr)
    genes = features["gene_name"].astype(str).to_numpy()
    lib_size = np.asarray(expr.sum(axis=0)).ravel()
    norm = expr @ sparse.diags(10000.0 / np.maximum(lib_size, 1))
    log_expr = norm.copy()
    log_expr.data = np.log1p(log_expr.data)

    cell_table = dge_cells[["CellBC", "treatment_group", "total_assigned_treatment_umi", "top_treatment", "top_treatment_umi", "top_treatment_fraction"]].copy()
    cell_table["gex_umi"] = lib_size
    cells_used_path = TABLES / "dc_dge_cells_used.csv"
    cell_table.to_csv(cells_used_path, index=False)

    results = []
    for treatment in contrasts:
        idx_t = np.where(cell_table["treatment_group"].to_numpy() == treatment)[0]
        idx_c = np.where(cell_table["treatment_group"].to_numpy() == "no_treatment")[0]
        pct_t = np.asarray((expr[:, idx_t] > 0).mean(axis=1)).ravel()
        pct_c = np.asarray((expr[:, idx_c] > 0).mean(axis=1)).ravel()
        mean_norm_t = np.asarray(norm[:, idx_t].mean(axis=1)).ravel()
        mean_norm_c = np.asarray(norm[:, idx_c].mean(axis=1)).ravel()
        mean_log_t, var_log_t = sparse_row_mean_var(log_expr, idx_t)
        mean_log_c, var_log_c = sparse_row_mean_var(log_expr, idx_c)
        total_counts = np.asarray(expr[:, np.r_[idx_t, idx_c]].sum(axis=1)).ravel()
        tested = (total_counts >= min_total_counts) & ((pct_t >= min_pct) | (pct_c >= min_pct))
        se = np.sqrt(var_log_t / max(len(idx_t), 1) + var_log_c / max(len(idx_c), 1))
        t_stat = np.divide(mean_log_t - mean_log_c, se, out=np.zeros_like(se), where=se > 0)
        denom = ((var_log_t / max(len(idx_t), 1)) ** 2 / max(len(idx_t) - 1, 1)) + ((var_log_c / max(len(idx_c), 1)) ** 2 / max(len(idx_c) - 1, 1))
        df = np.divide((var_log_t / max(len(idx_t), 1) + var_log_c / max(len(idx_c), 1)) ** 2, denom, out=np.full_like(denom, np.nan), where=denom > 0)
        p_value = 2 * stats.t.sf(np.abs(t_stat), df)
        p_value[~np.isfinite(p_value)] = 1.0
        padj = np.full_like(p_value, np.nan)
        padj[tested] = bh_adjust(p_value[tested])
        pooled = np.sqrt(((len(idx_t) - 1) * var_log_t + (len(idx_c) - 1) * var_log_c) / max(len(idx_t) + len(idx_c) - 2, 1))
        cohen = np.divide(mean_log_t - mean_log_c, pooled, out=np.zeros_like(pooled), where=pooled > 0)
        results.append(
            pd.DataFrame(
                {
                    "contrast": f"{treatment}_vs_no_treatment",
                    "treatment": treatment,
                    "control": "no_treatment",
                    "gene": genes,
                    "total_counts": total_counts,
                    "pct_treatment": pct_t,
                    "pct_control": pct_c,
                    "mean_norm_treatment": mean_norm_t,
                    "mean_norm_control": mean_norm_c,
                    "mean_log1p_treatment": mean_log_t,
                    "mean_log1p_control": mean_log_c,
                    "avg_log1p_diff": mean_log_t - mean_log_c,
                    "log2fc_mean_norm": np.log2((mean_norm_t + pseudocount) / (mean_norm_c + pseudocount)),
                    "welch_t": t_stat,
                    "welch_df": df,
                    "cohen_d_log1p": cohen,
                    "p_value": p_value,
                    "tested": tested,
                    "padj_bh": padj,
                    "n_treatment_cells": len(idx_t),
                    "n_control_cells": len(idx_c),
                }
            )
        )
    dge = pd.concat(results, ignore_index=True)
    dge_path = TABLES / "dc_dge_welch_results.csv.gz"
    dge.to_csv(dge_path, index=False)

    program_rows = []
    tested_dge = dge[dge["tested"]].copy()
    tested_dge["rank_score"] = np.sign(tested_dge["log2fc_mean_norm"]) * -np.log10(np.maximum(tested_dge["padj_bh"].fillna(1).to_numpy(float), 1e-300))
    for contrast, dt in tested_dge.groupby("contrast"):
        for program, program_genes in DC_PROGRAMS.items():
            sub = dt[dt["gene"].isin(program_genes)]
            rest = dt[~dt["gene"].isin(program_genes)]
            p_up = stats.mannwhitneyu(sub["rank_score"], rest["rank_score"], alternative="greater", method="asymptotic").pvalue if len(sub) >= 3 and len(rest) >= 3 else math.nan
            program_rows.append(
                {
                    "contrast": contrast,
                    "treatment": dt["treatment"].iloc[0],
                    "program": program,
                    "n_program_genes_input": len(program_genes),
                    "n_program_genes_tested": int(len(sub)),
                    "median_log2fc": float(sub["log2fc_mean_norm"].median()) if len(sub) else math.nan,
                    "fraction_sig_up": float(((sub["padj_bh"] < 0.05) & (sub["log2fc_mean_norm"] > 0.25)).mean()) if len(sub) else math.nan,
                    "wilcox_enrichment_p_up": float(p_up),
                    "top_up_genes": ";".join(sub.sort_values(["padj_bh", "log2fc_mean_norm"], ascending=[True, False])["gene"].head(12)),
                }
            )
    program_summary = pd.DataFrame(program_rows)
    for contrast, idx in program_summary.groupby("contrast").groups.items():
        program_summary.loc[list(idx), "padj_up_bh"] = bh_adjust(program_summary.loc[list(idx), "wilcox_enrichment_p_up"].to_numpy(float))
    program_summary_path = TABLES / "dc_dge_program_summary.csv"
    program_summary.to_csv(program_summary_path, index=False)

    heatmap_genes = []
    for treatment in contrasts:
        sub = dge[
            (dge["contrast"] == f"{treatment}_vs_no_treatment")
            & dge["tested"]
            & (dge["padj_bh"] < 0.05)
            & (dge["log2fc_mean_norm"] > 0.5)
        ]
        heatmap_genes.extend(sub.sort_values(["padj_bh", "log2fc_mean_norm"], ascending=[True, False])["gene"].head(25).tolist())
    heatmap_genes = list(dict.fromkeys([g for g in heatmap_genes if isinstance(g, str)]))
    gene_idx = [int(np.where(genes == g)[0][0]) for g in heatmap_genes if np.any(genes == g)]
    mean_rows = []
    for group in ["no_treatment"] + contrasts:
        idx = np.where(cell_table["treatment_group"].to_numpy() == group)[0]
        means = np.asarray(log_expr[gene_idx, :][:, idx].mean(axis=1)).ravel() if len(gene_idx) and len(idx) else np.array([])
        mean_rows.extend({"gene": gene, "treatment_group": group, "mean_log1p": float(value)} for gene, value in zip(heatmap_genes, means))
    heatmap_source = pd.DataFrame(mean_rows)
    if len(heatmap_source):
        heatmap_source["z"] = heatmap_source.groupby("gene")["mean_log1p"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) else 1.0))
    heatmap_source_path = TABLES / "dc_dge_top_gene_heatmap_source.csv"
    heatmap_source.to_csv(heatmap_source_path, index=False)

    primary_heatmap_summary = plot_dc_dge_top_gene_heatmap_primary(dge, contrasts, manifest)
    plot_dc_dge_outputs(dge, program_summary, heatmap_source, contrasts, heatmap_genes, manifest)
    summary = {
        "analysis": "dc_treatment_dge_90pct_purity_assignment",
        "treatment_purity_threshold": purity_threshold,
        "assignment_rule": "Assign dendritic cells when at least 90% of assigned DCBC UMI comes from a single treatment.",
        "min_pct_gene_filter": min_pct,
        "min_total_counts_gene_filter": min_total_counts,
        "contrasts": contrasts,
        "n_dge_cells": int(len(cell_table)),
        "n_dge_cells_by_group": {str(k): int(v) for k, v in cell_table["treatment_group"].value_counts().to_dict().items()},
        "n_dge_genes_unique_symbols": int(len(genes)),
        "n_tested_rows": int(dge["tested"].sum()),
        "n_heatmap_genes": int(len(heatmap_genes)),
        "dge_method": "Per-cell vectorized Welch test on log1p(CP10k) expression with BH adjustment; effect size is log2 fold-change of mean CP10k expression.",
        "assignment_table": str(assignment_path),
        "dge_results_table": str(dge_path),
        "program_summary_table": str(program_summary_path),
        "top_gene_heatmap_source": str(heatmap_source_path),
        "primary_top_gene_heatmap": primary_heatmap_summary,
    }
    write_json(summary, QC / "dc_treatment_dge_summary.json")
    return {"dc_treatment_dge": summary}


def plot_dc_dge_outputs(
    dge: pd.DataFrame,
    program_summary: pd.DataFrame,
    heatmap_source: pd.DataFrame,
    contrasts: list[str],
    heatmap_genes: list[str],
    manifest: list[dict[str, str]],
) -> None:
    heatmap_set = set(heatmap_genes)
    for treatment in contrasts:
        sub = dge[(dge["contrast"] == f"{treatment}_vs_no_treatment") & dge["tested"]].copy()
        sub["neglog10_padj"] = -np.log10(np.maximum(sub["padj_bh"].fillna(1).to_numpy(float), 1e-300))
        sub["in_top_gene_heatmap"] = sub["gene"].isin(heatmap_set)
        sub["sig"] = (sub["padj_bh"] < 0.05) & (sub["log2fc_mean_norm"].abs() > 0.5)
        volcano_source = TABLES / f"dc_dge_volcano_{treatment}_vs_no_treatment_source.csv"
        sub.to_csv(volcano_source, index=False)

        fig, ax = plt.subplots(figsize=(4.1, 3.1))
        other = sub[~sub["sig"]]
        sig = sub[sub["sig"] & ~sub["in_top_gene_heatmap"]]
        highlighted = sub[sub["sig"] & sub["in_top_gene_heatmap"]]
        ax.scatter(other["log2fc_mean_norm"], other["neglog10_padj"], s=2, color="#cfcfcf", alpha=0.45, linewidths=0)
        ax.scatter(sig["log2fc_mean_norm"], sig["neglog10_padj"], s=4, color="#ed8590", alpha=0.75, linewidths=0)
        ax.scatter(highlighted["log2fc_mean_norm"], highlighted["neglog10_padj"], s=10, color="#c83f58", alpha=0.95, edgecolors="#333333", linewidths=0.15)
        for _, row in highlighted.sort_values(["padj_bh", "log2fc_mean_norm"], ascending=[True, False]).head(15).iterrows():
            ax.text(row["log2fc_mean_norm"], row["neglog10_padj"], row["gene"], fontsize=5.5, color="#222222")
        ax.axvline(-0.5, color="#bbbbbb", linewidth=0.3)
        ax.axvline(0.5, color="#bbbbbb", linewidth=0.3)
        ax.axhline(-math.log10(0.05), color="#bbbbbb", linewidth=0.3)
        ax.set_xlabel("log2 fold-change of mean normalized expression")
        ax.set_ylabel("-log10 BH-adjusted p")
        ax.set_title(f"DC DGE: {treatment} vs no_treatment")
        files = save_dual(fig, f"dc_dge_volcano_{treatment}_vs_no_treatment")
        manifest.append({"figure": f"DC DGE volcano {treatment} vs no_treatment", "source_table": str(volcano_source), **files, "script_function": "run_dc_treatment_dge"})

    fig, ax = plt.subplots(figsize=(3.51, 3.0))
    programs = list(reversed(list(DC_PROGRAMS)))
    y = np.arange(len(programs))
    width = 0.34
    dge_program_colors = {"LPS": "#f2aa59", "IFNg": "#6dead4"}
    for offset, treatment in zip([-width / 2, width / 2], contrasts):
        sub = program_summary[program_summary["treatment"] == treatment].set_index("program").reindex(programs)
        ax.barh(y + offset, sub["median_log2fc"], height=width, color=dge_program_colors.get(treatment, "#cfcfcf"), edgecolor="white", linewidth=0.2, label=treatment)
    ax.axvline(0, color="#777777", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(programs)
    ax.set_xlabel("Median program log2FC vs no_treatment")
    ax.set_title("DC gene-program shifts")
    ax.legend(fontsize=6)
    program_source = TABLES / "dc_dge_program_summary.csv"
    files = save_dual(fig, "dc_dge_program_summary")
    manifest.append({"figure": "DC DGE program summary", "source_table": str(program_source), **files, "script_function": "run_dc_treatment_dge"})

    heatmap_source_path = TABLES / "dc_dge_top_gene_heatmap_source.csv"
    if len(heatmap_source):
        pivot = heatmap_source.pivot(index="gene", columns="treatment_group", values="z").reindex(index=list(reversed(heatmap_genes)), columns=["no_treatment"] + contrasts)
        fig_h = max(3.2, 0.095 * len(pivot))
        fig, ax = plt.subplots(figsize=(2.4, fig_h))
        cmap = LinearSegmentedColormap.from_list("dc_dge_heatmap", [HEATMAP_LOW, HEATMAP_MID, HEATMAP_HIGH])
        im = ax.imshow(np.clip(pivot.to_numpy(float), -2.5, 2.5), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=5.5)
        ax.set_title("Top treatment-up DC genes")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="z")
    else:
        fig, ax = plt.subplots(figsize=(2.4, 1.8))
        ax.text(0.5, 0.5, "No significant genes", ha="center", va="center", fontsize=7)
        ax.set_axis_off()
    files = save_dual(fig, "dc_dge_top_gene_heatmap")
    manifest.append({"figure": "DC DGE top gene heatmap", "source_table": str(heatmap_source_path), **files, "script_function": "run_dc_treatment_dge"})


def umap_plot_sample(umap: pd.DataFrame, max_points_by_label: dict[str, int], seed: int = 475, bins: int = 180) -> pd.DataFrame:
    """Thin dense embedding regions while preserving sparse UMAP structure."""
    sampled_indices = []
    rng = np.random.default_rng(seed)
    x = umap["UMAP-1"].to_numpy(float)
    y = umap["UMAP-2"].to_numpy(float)
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_bin = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, hist.shape[0] - 1)
    y_bin = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, hist.shape[1] - 1)
    density = hist[x_bin, y_bin]
    working = umap.copy()
    working["local_umap_bin_density"] = density
    for label, max_points in max_points_by_label.items():
        sub = working[working["plot_label"] == label]
        if len(sub) <= max_points:
            sampled_indices.extend(sub.index.tolist())
            continue
        weights = 1.0 / np.sqrt(np.maximum(sub["local_umap_bin_density"].to_numpy(float), 1.0))
        weights = weights / weights.sum()
        sampled_indices.extend(rng.choice(sub.index.to_numpy(), size=max_points, replace=False, p=weights).tolist())
    sampled = working.loc[sampled_indices].copy()
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def plot_standard_umap(inputs: dict[str, Any], manifest: list[dict[str, str]]) -> dict[str, Any]:
    umap = inputs["umap"].copy()
    out_table = TABLES / "umap_standard_source.csv"
    umap.to_csv(out_table, index=False)
    max_points_by_label = {"dendritic cell": 5200, "OTI T cell": 5200, "C57BL/6 T cell": 3600}
    plot_umap = umap_plot_sample(umap, max_points_by_label=max_points_by_label, seed=475)
    plot_table = TABLES / "umap_standard_plot_points.csv"
    plot_umap.to_csv(plot_table, index=False)
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    order = ["dendritic cell", "C57BL/6 T cell", "OTI T cell"]
    colors = {"dendritic cell": "#ed8590", "OTI T cell": OTI_BLUE, "C57BL/6 T cell": C57_GREY}
    plot_order = sorted(order, key=lambda label: int((plot_umap["plot_label"] == label).sum()), reverse=True)
    handles: dict[str, Any] = {}
    for label in plot_order:
        sub = plot_umap[plot_umap["plot_label"] == label]
        handles[label] = ax.scatter(
            sub["UMAP-1"],
            sub["UMAP-2"],
            s=1.15,
            alpha=0.82,
            linewidths=0,
            color=colors[label],
            label=label,
            rasterized=False,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend([handles[label] for label in order], order, markerscale=5.4, fontsize=5, loc="best")
    files = save_dual(fig, "umap_standard_cell_types")
    manifest.append({"figure": "UMAP standard cell types", "source_table": str(plot_table), **files})
    return {
        "umap_standard_total_cells": int(len(umap)),
        "umap_standard_plotted_cells": int(len(plot_umap)),
        "umap_standard_plotted_by_label": {str(k): int(v) for k, v in plot_umap["plot_label"].value_counts().to_dict().items()},
    }


def plot_interaction_umap(inputs: dict[str, Any], edges: pd.DataFrame, manifest: list[dict[str, str]], rng: np.random.Generator) -> dict[str, Any]:
    umap = inputs["umap"].copy()
    dc_meta = inputs["dc_meta"].copy()
    t_cells = set(inputs["t_meta"]["CellBC"].astype(str))
    coord = umap.set_index("CellBC")[["UMAP-1", "UMAP-2"]].to_dict("index")

    donor_dc = dc_meta[(dc_meta["top_DCBC"].astype(str).ne("")) & (pd.to_numeric(dc_meta["top_DCBC_fraction"], errors="coerce") > 0.95)].copy()
    donor_dc["top_DCBC_UMI"] = pd.to_numeric(donor_dc["top_DCBC_UMI"], errors="coerce").fillna(0)
    donor_dc = donor_dc.sort_values("top_DCBC_UMI", ascending=False).drop_duplicates("top_DCBC")
    dcbc_to_donor = dict(zip(donor_dc["top_DCBC"].astype(str), donor_dc["CellBC"].astype(str)))

    t_edges = edges[(edges["cell_class"] == "t_cell") & (edges["assigned_dcbc"]) & (edges["CellBC"].isin(t_cells))].copy()
    t_edges["donor_CellBC"] = t_edges["DCBC"].map(dcbc_to_donor)
    links = t_edges[t_edges["donor_CellBC"].fillna("").ne("")].copy()
    links = links[links["CellBC"].isin(coord) & links["donor_CellBC"].isin(coord)].copy()
    initial_min_umi = 5
    links = links[links["UMI"] >= initial_min_umi].copy()
    used_min_umi = initial_min_umi
    capped = False
    if len(links) > 2500:
        used_min_umi = max(initial_min_umi, int(np.percentile(links["UMI"], 90)))
        links = links[links["UMI"] >= used_min_umi].copy()
    if len(links) > 2500:
        links = links.sort_values("UMI", ascending=False).head(2500).copy()
        capped = True

    links["donor_UMAP_1"] = links["donor_CellBC"].map(lambda bc: coord[bc]["UMAP-1"])
    links["donor_UMAP_2"] = links["donor_CellBC"].map(lambda bc: coord[bc]["UMAP-2"])
    links["recipient_UMAP_1"] = links["CellBC"].map(lambda bc: coord[bc]["UMAP-1"])
    links["recipient_UMAP_2"] = links["CellBC"].map(lambda bc: coord[bc]["UMAP-2"])
    out_table = TABLES / "umap_single_cell_interaction_links.csv"
    links.to_csv(out_table, index=False)

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    background_umap = umap_plot_sample(
        umap,
        max_points_by_label={"dendritic cell": 4200, "OTI T cell": 4200, "C57BL/6 T cell": 2800},
        seed=476,
    )
    background_table = TABLES / "umap_single_cell_interaction_background_plot_points.csv"
    background_umap.to_csv(background_table, index=False)
    ax.scatter(background_umap["UMAP-1"], background_umap["UMAP-2"], s=0.85, color="#d9d9d9", alpha=0.24, linewidths=0, rasterized=False, zorder=0)
    if not links.empty:
        umi_min = float(links["UMI"].min())
        umi_max = float(links["UMI"].max())
        for row in links.itertuples(index=False):
            frac = (float(row.UMI) - umi_min) / max(umi_max - umi_min, 1.0)
            alpha = 0.06 + 0.34 * frac
            ax.plot([row.donor_UMAP_1, row.recipient_UMAP_1], [row.donor_UMAP_2, row.recipient_UMAP_2], color="#ed8590", alpha=alpha, linewidth=0.35, zorder=1)
        donor_pts = pd.DataFrame([coord[bc] for bc in sorted(set(links["donor_CellBC"]))])
        recipient_pts = pd.DataFrame([coord[bc] for bc in sorted(set(links["CellBC"]))])
        ax.scatter(donor_pts["UMAP-1"], donor_pts["UMAP-2"], s=1.4, color="#ed8590", alpha=0.72, label="DCBC donor DC", linewidths=0, zorder=3)
        ax.scatter(recipient_pts["UMAP-1"], recipient_pts["UMAP-2"], s=1.1, color="#a68ff8", alpha=0.72, label="T cell recipient", linewidths=0, zorder=2)
        grad_ax = ax.inset_axes([0.59, 0.075, 0.28, 0.035])
        rgba = np.array(matplotlib.colors.to_rgba("#ed8590"))
        gradient = np.ones((1, 128, 4), dtype=float)
        gradient[:, :, :3] = rgba[:3]
        gradient[:, :, 3] = np.linspace(0.06, 0.40, 128)
        grad_ax.imshow(gradient, aspect="auto", extent=[umi_min, umi_max, 0, 1])
        grad_ax.set_yticks([])
        grad_ax.set_xticks([umi_min, umi_max])
        grad_ax.set_xticklabels([f"{umi_min:.0f}", f"{umi_max:.0f}"], fontsize=4.8)
        grad_ax.set_title("Link opacity (UMI)", fontsize=5, pad=1)
        for spine in grad_ax.spines.values():
            spine.set_visible(False)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2, fontsize=5, loc="best")
    ax.set_title(f"DCBC transfer links, UMI >= {used_min_umi}", fontsize=7)
    files = save_dual(fig, "umap_single_cell_interaction_map")
    manifest.append({"figure": "UMAP single-cell interaction map", "source_table": str(out_table), **files})
    return {
        "interaction_umap_links": int(len(links)),
        "interaction_umap_min_umi": int(used_min_umi),
        "interaction_umap_capped_to_top_2500": capped,
        "interaction_umap_background_plotted_cells": int(len(background_umap)),
    }


def t_cell_dcbc_metrics(inputs: dict[str, Any], edges: pd.DataFrame) -> pd.DataFrame:
    t_meta = inputs["t_meta"].copy()
    t_edges = edges[(edges["cell_class"] == "t_cell") & (edges["assigned_dcbc"])].copy()
    unique_counts = t_edges.groupby("CellBC")["DCBC"].nunique()
    total_umi = t_edges.groupby("CellBC")["UMI"].sum()
    out = t_meta[["CellBC", "t_cell_type"]].copy()
    out["unique_dcbc"] = out["CellBC"].map(unique_counts).fillna(0).astype(int)
    out["total_dcbc_umi"] = out["CellBC"].map(total_umi).fillna(0).astype(int)
    out.to_csv(TABLES / "t_cell_dcbc_pickup_metrics.csv", index=False)
    return out


def plot_pickup_multiplicity(metrics: pd.DataFrame, manifest: list[dict[str, str]], rng: np.random.Generator) -> None:
    max_count = int(metrics["unique_dcbc"].max())
    rows = []
    for t_type in ["OTI", "C57BL6"]:
        sub = metrics[metrics["t_cell_type"] == t_type]
        denom = max(len(sub), 1)
        counts = sub["unique_dcbc"].value_counts()
        for count in range(max_count + 1):
            rows.append({"t_cell_type": t_type, "unique_dcbc": count, "fraction_t_cells": float(counts.get(count, 0) / denom), "n_t_cells": int(counts.get(count, 0))})
    df = pd.DataFrame(rows)
    out_table = TABLES / "unique_dcbc_pickup_multiplicity_source.csv"
    df.to_csv(out_table, index=False)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.12 * (max_count + 1) + 1.4), 2.3))
    width = 0.42
    x = np.arange(max_count + 1)
    for offset, t_type, color in [(-width / 2, "OTI", OTI_BLUE), (width / 2, "C57BL6", C57_GREY)]:
        sub = df[df["t_cell_type"] == t_type].set_index("unique_dcbc")
        ax.bar(x + offset, [sub.loc[i, "fraction_t_cells"] for i in x], width=width, color=color, linewidth=0, label=t_type)
    ax.set_xlabel("Unique DCBCs picked up")
    ax.set_ylabel("Fraction of T cells")
    ax.set_xticks(x[:: max(1, math.ceil(len(x) / 18))])
    ax.legend(fontsize=6)
    files = save_dual(fig, "unique_dcbc_pickup_multiplicity_bar")
    manifest.append({"figure": "Unique DCBC pickup multiplicity", "source_table": str(out_table), **files})

    dot_source = metrics.copy()
    dot_source["log10_total_dcbc_umi_plus1"] = np.log10(dot_source["total_dcbc_umi"] + 1)
    dot_table = TABLES / "unique_dcbc_pickup_single_cell_dot_source.csv"
    dot_source.to_csv(dot_table, index=False)
    fig, ax = plt.subplots(figsize=(2.25, 2.7))
    cmap = LinearSegmentedColormap.from_list("umi_blue_red", ["#cfeaff", "#84c7ff", "#4e9bd7", "#ed8590", "#c83f58"])
    for x0, t_type in enumerate(["OTI", "C57BL6"]):
        sub = dot_source[dot_source["t_cell_type"] == t_type]
        jitter = rng.normal(0, 0.055, len(sub))
        ax.scatter(
            np.full(len(sub), x0) + jitter,
            sub["unique_dcbc"],
            c=sub["log10_total_dcbc_umi_plus1"],
            cmap=cmap,
            s=6,
            alpha=0.62,
            linewidths=0,
            rasterized=False,
        )
        if len(sub):
            median_unique = float(np.median(sub["unique_dcbc"].to_numpy(dtype=float)))
            ax.plot([x0 - 0.24, x0 + 0.24], [median_unique, median_unique], color="#ffffff", linewidth=2.0, solid_capstyle="round", zorder=5)
            ax.plot([x0 - 0.24, x0 + 0.24], [median_unique, median_unique], color="#222222", linewidth=1.0, solid_capstyle="round", zorder=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["OTI", "C57BL/6"])
    ax.set_ylabel("Unique DCBCs picked up")
    ax.set_xlim(-0.45, 1.45)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=float(dot_source["log10_total_dcbc_umi_plus1"].min()), vmax=float(dot_source["log10_total_dcbc_umi_plus1"].max())))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label("log10(UMI + 1)", fontsize=6)
    files = save_dual(fig, "unique_dcbc_pickup_single_cell_dot")
    manifest.append({"figure": "Unique DCBC pickup single-cell dot", "source_table": str(dot_table), **files})


def condition_bubble_inputs(inputs: dict[str, Any], edges: pd.DataFrame, peptide_order: list[str], treatment_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    t_meta = inputs["t_meta"][["CellBC", "t_cell_type"]].copy()
    t_type = dict(zip(t_meta["CellBC"], t_meta["t_cell_type"]))
    dc_edges = edges[(edges["cell_class"] == "dendritic_cell") & (edges["assigned_dcbc"])].copy()
    donor_denoms = (
        dc_edges.groupby(["AssignedPeptideBC_Name", "AssignedTreatment"])["DCBC"]
        .nunique()
        .rename("global_unique_dcbc")
        .reset_index()
    )
    denom_map = {(r.AssignedPeptideBC_Name, r.AssignedTreatment): int(r.global_unique_dcbc) for r in donor_denoms.itertuples(index=False)}
    denom_rows = []
    for peptide in peptide_order:
        for treatment in treatment_order:
            denom_rows.append({"PeptideBC_Name": peptide, "AssignedTreatment": treatment, "global_unique_dcbc": denom_map.get((peptide, treatment), 0)})
    denom_full = pd.DataFrame(denom_rows)
    denom_full.to_csv(TABLES / "bubble_condition_global_unique_dcbc.csv", index=False)

    t_edges = edges[(edges["cell_class"] == "t_cell") & (edges["assigned_dcbc"])].copy()
    t_edges["t_cell_type"] = t_edges["CellBC"].map(t_type)
    cell_condition = (
        t_edges.groupby(["CellBC", "t_cell_type", "AssignedPeptideBC_Name", "AssignedTreatment"], as_index=False)["UMI"]
        .sum()
        .rename(columns={"UMI": "condition_umi"})
    )
    qualifying = cell_condition[cell_condition["condition_umi"] >= 2].copy()
    rows = []
    for t_cell_type in ["OTI", "C57BL6"]:
        for peptide_i, peptide in enumerate(peptide_order):
            for treatment_i, treatment in enumerate(treatment_order):
                vals = qualifying.loc[
                    (qualifying["t_cell_type"] == t_cell_type)
                    & (qualifying["AssignedPeptideBC_Name"] == peptide)
                    & (qualifying["AssignedTreatment"] == treatment),
                    "condition_umi",
                ].to_numpy(dtype=float)
                denom = denom_map.get((peptide, treatment), 0)
                rows.append(
                    {
                        "t_cell_type": t_cell_type,
                        "PeptideBC_Name": peptide,
                        "AssignedTreatment": treatment,
                        "peptide_order": peptide_i,
                        "treatment_order": treatment_i,
                        "qualifying_cells": int(len(vals)),
                        "global_unique_dcbc": int(denom),
                        "cells_per_global_unique_dcbc": float(len(vals) / denom) if denom else 0.0,
                        "geomean_condition_umi": float(np.exp(np.mean(np.log(vals)))) if len(vals) else 0.0,
                    }
                )
    bubble = pd.DataFrame(rows)
    bubble.to_csv(TABLES / "bubble_peptide_treatment_source.csv", index=False)
    qualifying.to_csv(TABLES / "bubble_peptide_treatment_qualifying_cell_conditions.csv", index=False)
    return bubble, denom_full


def plot_condition_bubbles(bubble: pd.DataFrame, peptide_order: list[str], treatment_order: list[str], manifest: list[dict[str, str]]) -> None:
    positive = bubble.loc[bubble["geomean_condition_umi"] > 0, "geomean_condition_umi"].to_numpy(float)
    vmin = float(np.percentile(positive, 5)) if len(positive) else 0.0
    vmax = float(np.percentile(positive, 90)) if len(positive) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    size_max = max(float(bubble["cells_per_global_unique_dcbc"].max()), 1e-12)
    def bubble_size(values: np.ndarray | float) -> np.ndarray | float:
        return 18 + (np.asarray(values, dtype=float) / size_max) * 230

    size_legend_values = bubble.loc[bubble["cells_per_global_unique_dcbc"] > 0, "cells_per_global_unique_dcbc"].to_numpy(float)
    size_legend_values = np.unique(np.round(np.percentile(size_legend_values, [25, 50, 90]), 2)) if len(size_legend_values) else np.array([])
    cmap = LinearSegmentedColormap.from_list("condition_red", ["#f7fbff", "#f9d6d6", "#ed8590", "#c83f58"])
    for t_cell_type in ["OTI", "C57BL6"]:
        sub = bubble[bubble["t_cell_type"] == t_cell_type].copy()
        fig, ax = plt.subplots(figsize=(2.9, 4.6))
        ax.set_axisbelow(True)
        plot_sub = sub[sub["geomean_condition_umi"] > 0].copy()
        if not plot_sub.empty:
            sizes = bubble_size(plot_sub["cells_per_global_unique_dcbc"].to_numpy(float))
            colors = np.clip(plot_sub["geomean_condition_umi"].to_numpy(float), vmin, vmax)
            ax.scatter(plot_sub["treatment_order"], plot_sub["peptide_order"], s=sizes, c=colors, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="#7a2f36", linewidths=0.35, zorder=3)
        ax.set_xticks(range(len(treatment_order)))
        ax.set_xticklabels(treatment_order, rotation=35, ha="right")
        ax.set_yticks(range(len(peptide_order)))
        ax.set_yticklabels(peptide_order)
        ax.set_ylim(len(peptide_order) - 0.5, -0.5)
        ax.set_xlim(-0.5, len(treatment_order) - 0.5)
        ax.set_title(f"{t_cell_type} transfer")
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Peptide")
        ax.grid(True, color="#e8e8e8", linewidth=0.35, zorder=0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.055, pad=0.04)
        cbar.set_label("Geomean UMI", fontsize=6)
        if len(size_legend_values):
            handles = [
                ax.scatter([], [], s=float(bubble_size(value)), facecolor="#f9d6d6", edgecolor="#7a2f36", linewidths=0.35)
                for value in size_legend_values
            ]
            ax.legend(
                handles,
                [f"{value:g}" for value in size_legend_values],
                title="Cells / donor DCBC",
                fontsize=5,
                title_fontsize=5,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=len(size_legend_values),
                columnspacing=1.0,
                handletextpad=0.7,
            )
        files = save_dual(fig, f"bubble_peptide_treatment_{t_cell_type}")
        manifest.append({"figure": f"Bubble peptide+treatment matrix {t_cell_type}", "source_table": str(TABLES / "bubble_peptide_treatment_source.csv"), **files})


def peptide_umi_lfc(edges: pd.DataFrame, peptide_order: list[str], manifest: list[dict[str, str]]) -> dict[str, Any]:
    assigned = edges[edges["assigned_dcbc"]].copy()
    t = assigned[assigned["cell_class"] == "t_cell"].groupby("AssignedPeptideBC_Name")["UMI"].sum()
    dc = assigned[assigned["cell_class"] == "dendritic_cell"].groupby("AssignedPeptideBC_Name")["UMI"].sum()
    t_total = float(t.sum())
    dc_total = float(dc.sum())
    pseudo = 1e-9
    rows = []
    for peptide in peptide_order:
        t_prop = float(t.get(peptide, 0) / t_total) if t_total else 0.0
        dc_prop = float(dc.get(peptide, 0) / dc_total) if dc_total else 0.0
        rows.append({"PeptideBC_Name": peptide, "t_cell_umi": int(t.get(peptide, 0)), "dc_umi": int(dc.get(peptide, 0)), "t_cell_proportion": t_prop, "dc_proportion": dc_prop, "log2_lfc_t_over_dc": float(np.log2((t_prop + pseudo) / (dc_prop + pseudo)))})
    lfc = pd.DataFrame(rows)
    lfc_path = ROOT / "data_external" / "lfc_bulk.csv"
    out_table = TABLES / "peptide_umi_normalized_vs_bulk_lfc_source.csv"
    if not lfc_path.exists():
        lfc.to_csv(out_table, index=False)
        (QC / "peptide_umi_normalized_vs_bulk_lfc_SKIPPED.txt").write_text(f"Missing required external file: {lfc_path}\n", encoding="utf-8")
        return {"peptide_lfc_bulk_plot": "skipped_missing_data_external_lfc_bulk_csv"}

    bulk = pd.read_csv(lfc_path)
    peptide_col = next((c for c in bulk.columns if c.lower() in {"peptide", "peptidebc_name", "name"} or "peptide" in c.lower()), bulk.columns[0])
    lfc_col = next((c for c in bulk.columns if "lfc" in c.lower() or "log2" in c.lower()), None)
    if lfc_col is None:
        numeric_candidates = []
        for col in bulk.columns:
            if col == peptide_col:
                continue
            values = pd.to_numeric(bulk[col], errors="coerce")
            if values.notna().sum() > 0:
                numeric_candidates.append(col)
        lfc_col = numeric_candidates[0] if numeric_candidates else None
    if lfc_col is None:
        lfc.to_csv(out_table, index=False)
        (QC / "peptide_umi_normalized_vs_bulk_lfc_SKIPPED.txt").write_text(f"No numeric bulk LFC column found in {lfc_path}. Columns: {list(bulk.columns)}\n", encoding="utf-8")
        return {"peptide_lfc_bulk_plot": "skipped_missing_lfc_column"}
    bulk[lfc_col] = pd.to_numeric(bulk[lfc_col], errors="coerce")
    merged = lfc.merge(bulk[[peptide_col, lfc_col]].rename(columns={peptide_col: "PeptideBC_Name", lfc_col: "bulk_log2_lfc"}), on="PeptideBC_Name", how="left")
    control_peptides = ["MCMV", "LCMV", "CATNB", "TB"]
    merged["is_control_center_peptide"] = merged["PeptideBC_Name"].isin(control_peptides)
    bulk_center = float(merged.loc[merged["is_control_center_peptide"], "bulk_log2_lfc"].mean(skipna=True))
    barcode_center = float(merged.loc[merged["is_control_center_peptide"], "log2_lfc_t_over_dc"].mean(skipna=True))
    if not np.isfinite(bulk_center):
        bulk_center = 0.0
    if not np.isfinite(barcode_center):
        barcode_center = 0.0
    merged["bulk_log2_lfc_centered"] = merged["bulk_log2_lfc"] - bulk_center
    merged["log2_lfc_t_over_dc_centered"] = merged["log2_lfc_t_over_dc"] - barcode_center
    merged["bulk_control_mean_subtracted"] = bulk_center
    merged["barcode_control_mean_subtracted"] = barcode_center
    merged.to_csv(out_table, index=False)
    plot_df = merged.dropna(subset=["bulk_log2_lfc_centered", "log2_lfc_t_over_dc_centered"]).copy()
    finite_mask = np.isfinite(plot_df["bulk_log2_lfc_centered"]) & np.isfinite(plot_df["log2_lfc_t_over_dc_centered"])
    plot_df = plot_df.loc[finite_mask]
    x = plot_df["bulk_log2_lfc_centered"].to_numpy(dtype=float)
    y = plot_df["log2_lfc_t_over_dc_centered"].to_numpy(dtype=float)
    correlation_stats: dict[str, Any] = {
        "n": int(len(plot_df)),
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
        "linear_regression_slope": None,
        "linear_regression_intercept": None,
        "linear_regression_r_squared": None,
        "linear_regression_p": None,
    }
    regression = None
    if len(plot_df) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        regression = stats.linregress(x, y)
        correlation_stats.update(
            {
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "linear_regression_slope": float(regression.slope),
                "linear_regression_intercept": float(regression.intercept),
                "linear_regression_r_squared": float(regression.rvalue**2),
                "linear_regression_p": float(regression.pvalue),
            }
        )
    (QC / "peptide_umi_normalized_vs_bulk_lfc_correlation.json").write_text(json.dumps(correlation_stats, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    if regression is not None:
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        line_y = regression.intercept + regression.slope * line_x
        ax.plot(line_x, line_y, color="#c83f58", linewidth=0.9, zorder=1)
    colors = np.where(plot_df["is_control_center_peptide"], CONTROL_GREY, SKY_BLUE)
    ax.scatter(plot_df["bulk_log2_lfc_centered"], plot_df["log2_lfc_t_over_dc_centered"], s=24, color=colors, edgecolors="none", linewidths=0, zorder=2)
    for row in plot_df.itertuples(index=False):
        ax.text(row.bulk_log2_lfc_centered, row.log2_lfc_t_over_dc_centered, row.PeptideBC_Name, fontsize=5, ha="left", va="bottom")
    ax.axhline(0, color="#777777", linewidth=0.5)
    ax.axvline(0, color="#777777", linewidth=0.5)
    if correlation_stats["pearson_r"] is not None:
        ax.text(
            0.03,
            0.97,
            f"Pearson r={correlation_stats['pearson_r']:.2f}, p={correlation_stats['pearson_p']:.2g}\n"
            f"Spearman rho={correlation_stats['spearman_rho']:.2f}, p={correlation_stats['spearman_p']:.2g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    ax.set_xlabel("Bulk screen log2 fold change, centered")
    ax.set_ylabel("T/DC barcode proportion log2 fold change, centered")
    files = save_dual(fig, "peptide_umi_normalized_vs_bulk_lfc")
    manifest.append({"figure": "Peptide UMI normalized vs bulk LFC", "source_table": str(out_table), **files})
    return {"peptide_lfc_bulk_plot": "generated", "bulk_lfc_file": str(lfc_path), "peptide_lfc_correlation": correlation_stats}


def load_10x_matrix(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], sparse.csc_matrix]:
    matrix_tar = (PUB / config["paths"]["filtered_feature_matrix"]).resolve()
    with tarfile.open(matrix_tar, "r:gz") as tar, tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tar.extractall(tmp, filter="data")
        files = list(tmp.rglob("*"))
        matrix_path = next(p for p in files if p.name == "matrix.mtx.gz")
        features_path = next(p for p in files if p.name in {"features.tsv.gz", "genes.tsv.gz"})
        barcodes_path = next(p for p in files if p.name == "barcodes.tsv.gz")
        matrix = scipy_io.mmread(matrix_path).tocsc()
        features = pd.read_csv(features_path, sep="\t", header=None, names=["gene_id", "gene_name", "feature_type"], compression="gzip")
        barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, names=["barcode"], compression="gzip")["barcode"].astype(str).tolist()
    return features, barcodes, matrix


def build_gene_index(features: pd.DataFrame) -> dict[str, list[int]]:
    gene_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in features.iterrows():
        if str(row.get("feature_type", "")) == "Gene Expression":
            gene_to_indices[str(row["gene_name"])].append(int(i))
    return gene_to_indices


def expression_by_group(matrix: sparse.csc_matrix, gene_to_indices: dict[str, list[int]], barcodes: list[str], panel: list[str], groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    library_size = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
    mean_rows = []
    positive_rows = []
    missing = []
    for gene in panel:
        if gene not in gene_to_indices:
            missing.append(gene)
            continue
        gene_idx = gene_to_indices[gene][0]
        mean_row: dict[str, Any] = {"gene": gene}
        positive_row: dict[str, Any] = {"gene": gene}
        for group_name, group_cells in groups.items():
            idx = np.array([barcode_to_idx[bc] for bc in group_cells if bc in barcode_to_idx], dtype=int)
            if len(idx) == 0:
                mean_row[group_name] = np.nan
                positive_row[group_name] = np.nan
                continue
            counts = matrix[gene_idx, idx].toarray().ravel().astype(np.float32)
            norm = np.log1p(counts / np.maximum(library_size[idx], 1.0) * 10000.0)
            mean_row[group_name] = float(norm.mean())
            positive = norm[counts > 0]
            positive_row[group_name] = float(positive.mean()) if len(positive) else 0.0
        mean_rows.append(mean_row)
        positive_rows.append(positive_row)
    return pd.DataFrame(mean_rows), pd.DataFrame(positive_rows), missing


def drop_blank_expression_rows(mean_df: pd.DataFrame, positive_df: pd.DataFrame | None, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    if mean_df.empty:
        return mean_df, positive_df, []
    values = np.nan_to_num(mean_df[cols].to_numpy(dtype=float), nan=0.0)
    keep = np.any(values != 0.0, axis=1)
    dropped = mean_df.loc[~keep, "gene"].astype(str).tolist()
    filtered_mean = mean_df.loc[keep].reset_index(drop=True)
    filtered_positive = positive_df.loc[keep].reset_index(drop=True) if positive_df is not None else None
    return filtered_mean, filtered_positive, dropped


def zscore_rows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = df.copy()
    values = z[cols].to_numpy(dtype=float)
    means = np.nanmean(values, axis=1, keepdims=True)
    stds = np.nanstd(values, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    z[cols] = np.nan_to_num((values - means) / stds, nan=0.0)
    return z


def clustered_heatmap(z_df: pd.DataFrame, cols: list[str], title: str, name: str, manifest: list[dict[str, str]], source_table: Path) -> None:
    x = z_df[cols].to_numpy(dtype=float)
    if len(z_df) > 2:
        linkage = hierarchy.linkage(np.nan_to_num(x), method="average", metric="euclidean")
        order = hierarchy.leaves_list(linkage)
    else:
        linkage = None
        order = np.arange(len(z_df))
    ordered = z_df.iloc[order].reset_index(drop=True)
    cmap = LinearSegmentedColormap.from_list("heat", ["#5d8af7", "#ffffff", "#ed8590"])
    fig = plt.figure(figsize=(max(2.8, 0.42 * len(cols) + 1.4), max(3.0, min(12.0, 0.055 * len(ordered) + 1.1))))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.42, 2.8], wspace=0.03)
    ax_d = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    if linkage is not None:
        hierarchy.dendrogram(linkage, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="#555555", ax=ax_d)
    ax_d.axis("off")
    im = ax_h.imshow(ordered[cols].to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap=cmap, vmin=-2, vmax=2)
    ax_h.set_xticks(range(len(cols)))
    ax_h.set_xticklabels(cols, rotation=45, ha="right", fontsize=6)
    ax_h.set_yticks([])
    ax_h.set_title(title, fontsize=8)
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
    cbar.set_label("Row z-score", fontsize=6)
    files = save_dual(fig, name)
    manifest.append({"figure": title, "source_table": str(source_table), **files})


def zscore_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return math.nan
    u = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def bootstrap_median_diff_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = 300) -> tuple[float, float]:
    if len(x) == 0 or len(y) == 0:
        return math.nan, math.nan
    bx = rng.choice(x, size=(n_boot, len(x)), replace=True)
    by = rng.choice(y, size=(n_boot, len(y)), replace=True)
    diffs = np.median(bx, axis=1) - np.median(by, axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def single_cell_lfc_peptide_order(peptide_order: list[str]) -> tuple[list[str], list[str], pd.DataFrame, str]:
    metric = "log2_lfc_t_over_dc_centered"
    lfc_path = TABLES / "peptide_umi_normalized_vs_bulk_lfc_source.csv"
    fallback_noncontrols = [p for p in peptide_order if p not in CONTROL_PEPTIDES]
    controls = [p for p in CONTROL_PEPTIDES if p in peptide_order]
    if not lfc_path.exists():
        order = fallback_noncontrols + controls
        table = pd.DataFrame({"PeptideBC_Name": order, "single_cell_lfc_metric": metric, "single_cell_lfc": np.nan})
        table["order_index"] = np.arange(len(order))
        table["order_class"] = ["noncontrol_config_order"] * len(fallback_noncontrols) + ["control_pinned_far_right"] * len(controls)
        return order, fallback_noncontrols, table, metric

    lfc = pd.read_csv(lfc_path)
    if metric not in lfc.columns:
        metric = "log2_lfc_t_over_dc"
    lfc_map = dict(zip(lfc["PeptideBC_Name"].astype(str), pd.to_numeric(lfc[metric], errors="coerce")))

    def order_value(peptide: str) -> float:
        value = float(lfc_map.get(peptide, -np.inf))
        return value if np.isfinite(value) else -np.inf

    noncontrols = sorted(fallback_noncontrols, key=lambda p: (order_value(p), p), reverse=True)
    order = noncontrols + controls
    table = pd.DataFrame(
        {
            "PeptideBC_Name": order,
            "single_cell_lfc_metric": metric,
            "single_cell_lfc": [float(lfc_map.get(p, np.nan)) for p in order],
            "order_index": np.arange(len(order)),
            "order_class": ["noncontrol_sorted_by_single_cell_lfc_desc"] * len(noncontrols) + ["control_pinned_far_right"] * len(controls),
            "lfc_source_table": str(lfc_path),
        }
    )
    return order, noncontrols, table, metric


def signature_group_orders(noncontrol_peptide_order: list[str]) -> dict[str, list[str]]:
    controls = [p for p in CONTROL_PEPTIDES if p not in noncontrol_peptide_order]
    effect_order = ["multi_peptide"] + list(noncontrol_peptide_order) + controls
    return {
        "signature_heatmap": effect_order + ["no_interaction"],
        "effect_ci": effect_order,
        "highlighted_gene_bubble": effect_order,
    }


def derive_oti_peptide_groups(t_meta: pd.DataFrame, peptide_order: list[str], heatmap_order: list[str], group_col: str = "peptide_group") -> pd.DataFrame:
    oti = t_meta[t_meta["t_cell_type"].eq("OTI")].copy()
    oti[group_col] = ""
    assigned = oti["AssignedPeptideBC_Name"].astype(str)
    oti.loc[assigned.isin(peptide_order), group_col] = assigned
    oti.loc[(oti[group_col].eq("")) & oti["interaction_state"].eq("no_interaction"), group_col] = "no_interaction"
    oti.loc[(oti[group_col].eq("")) & ~oti["interaction_state"].eq("no_interaction"), group_col] = "multi_peptide"
    oti[group_col] = pd.Categorical(oti[group_col], categories=heatmap_order, ordered=True)
    return oti[oti[group_col].notna()].copy()


def compute_t_cell_signature_scores(
    features: pd.DataFrame,
    barcodes: list[str],
    matrix: sparse.csc_matrix,
    t_meta: pd.DataFrame,
    peptide_order: list[str],
    heatmap_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oti = derive_oti_peptide_groups(t_meta, peptide_order, heatmap_order)
    barcode_to_col = {bc: i for i, bc in enumerate(barcodes)}
    oti = oti[oti["CellBC"].isin(barcode_to_col)].copy()
    oti["matrix_col"] = oti["CellBC"].map(barcode_to_col).astype(int)
    col_idx = oti["matrix_col"].to_numpy(dtype=int)
    library_size = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)

    gene_to_index: dict[str, int] = {}
    for idx, row in features.iterrows():
        if str(row.get("feature_type", "")) == "Gene Expression" and str(row["gene_name"]) not in gene_to_index:
            gene_to_index[str(row["gene_name"])] = int(idx)

    signature_rows = []
    gene_expr: dict[str, np.ndarray] = {}
    for signature, genes in T_CELL_SIGNATURES.items():
        present = []
        for gene in genes:
            if gene in gene_to_index and gene not in present:
                present.append(gene)
        signature_rows.append(
            {
                "signature": signature,
                "n_input_genes": len(dict.fromkeys(genes)),
                "n_present_genes": len(present),
                "present_genes": ";".join(present),
            }
        )
        for gene in present:
            if gene in gene_expr:
                continue
            counts = matrix[gene_to_index[gene], col_idx].toarray().ravel().astype(np.float32)
            gene_expr[gene] = np.log1p(counts / np.maximum(library_size[col_idx], 1.0) * 10000.0)

    scores = oti[["CellBC", "peptide_group", "total_barcode_umi", "interaction_state", "AssignedTreatment"]].copy()
    scores["peptide_group"] = scores["peptide_group"].astype(str)
    for signature, genes in T_CELL_SIGNATURES.items():
        present = [gene for gene in dict.fromkeys(genes) if gene in gene_expr]
        if not present:
            scores[signature] = np.nan
            continue
        z_gene = np.vstack([zscore_array(gene_expr[gene]) for gene in present])
        scores[signature] = np.nanmean(z_gene, axis=0)
    return scores, pd.DataFrame(signature_rows)


def summarize_t_cell_signature_groups(scores: pd.DataFrame, heatmap_order: list[str]) -> pd.DataFrame:
    rows = []
    for signature in T_CELL_SIGNATURES:
        for group in heatmap_order:
            vals = scores.loc[scores["peptide_group"].astype(str).eq(group), signature].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "signature": signature,
                    "peptide_group": group,
                    "n_cells": int(len(vals)),
                    "mean_score": float(np.mean(vals)),
                    "median_score": float(np.median(vals)),
                    "q25": float(np.percentile(vals, 25)),
                    "q75": float(np.percentile(vals, 75)),
                }
            )
    return pd.DataFrame(rows)


def t_cell_signature_pairwise_vs_no_interaction(scores: pd.DataFrame, heatmap_order: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(475)
    ref_vals_by_signature = {
        signature: scores.loc[scores["peptide_group"].astype(str).eq("no_interaction"), signature].dropna().to_numpy(float)
        for signature in T_CELL_SIGNATURES
    }
    rows = []
    for signature in T_CELL_SIGNATURES:
        ref_vals = ref_vals_by_signature[signature]
        for group in heatmap_order:
            if group == "no_interaction":
                continue
            vals = scores.loc[scores["peptide_group"].astype(str).eq(group), signature].dropna().to_numpy(float)
            if len(vals) < 5 or len(ref_vals) < 5:
                continue
            test = stats.mannwhitneyu(vals, ref_vals, alternative="two-sided", method="asymptotic")
            ci_low, ci_high = bootstrap_median_diff_ci(vals, ref_vals, rng)
            rows.append(
                {
                    "signature": signature,
                    "comparison": f"{group}_vs_no_interaction",
                    "peptide_group": group,
                    "reference": "no_interaction",
                    "n_group": int(len(vals)),
                    "n_reference": int(len(ref_vals)),
                    "median_group": float(np.median(vals)),
                    "median_reference": float(np.median(ref_vals)),
                    "median_diff": float(np.median(vals) - np.median(ref_vals)),
                    "median_diff_ci95_low": ci_low,
                    "median_diff_ci95_high": ci_high,
                    "rank_biserial": rank_biserial(vals, ref_vals),
                    "p_value": float(test.pvalue),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh_all"] = bh_adjust(out["p_value"].to_numpy(float))
    out["q_value_bh_by_signature"] = np.nan
    for signature, idx in out.groupby("signature").groups.items():
        out.loc[list(idx), "q_value_bh_by_signature"] = bh_adjust(out.loc[list(idx), "p_value"].to_numpy(float))
    return out


def plot_t_cell_signature_group_median_heatmap_ordered(
    group_summary: pd.DataFrame,
    signature_order: list[str],
    group_order: list[str],
    manifest: list[dict[str, str]],
) -> pd.DataFrame:
    pivot = group_summary.pivot(index="signature", columns="peptide_group", values="median_score").reindex(
        index=signature_order,
        columns=group_order,
    )
    z = pivot.apply(lambda row: (row - row.mean()) / (row.std(ddof=0) if row.std(ddof=0) else 1.0), axis=1)
    source_path = TABLES / "t_cell_signature_group_median_heatmap_zscore.csv"
    z.to_csv(source_path)

    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    cmap = LinearSegmentedColormap.from_list("signature_heatmap", [HEATMAP_LOW, HEATMAP_MID, HEATMAP_HIGH])
    im = ax.imshow(np.clip(z.to_numpy(float), -2.5, 2.5), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(z.columns)))
    ax.set_xticklabels(["multi" if group == "multi_peptide" else group for group in z.columns], rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels(z.index, fontsize=6)
    ax.set_title("OTI peptide-group T-cell signature medians", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, label="Row z-score")
    files = save_dual(fig, "t_cell_signature_group_median_heatmap")
    manifest.append({"figure": "T cell signature group median heatmap", "source_table": str(source_path), **files, "script_function": "plot_t_cell_signature_group_median_heatmap_ordered"})
    return z


def plot_tcell_signature_effect_ci_vs_no_interaction_ordered(
    pairwise_no_interaction: pd.DataFrame,
    signature_order: list[str],
    group_order: list[str],
    manifest: list[dict[str, str]],
    transfer_metric: pd.DataFrame,
) -> pd.DataFrame:
    source = pairwise_no_interaction[
        pairwise_no_interaction["signature"].isin(signature_order)
        & pairwise_no_interaction["peptide_group"].isin(group_order)
    ].copy()
    source["peptide_order"] = source["peptide_group"].map({group: i for i, group in enumerate(group_order)})
    source["signature_order"] = source["signature"].map({signature: i for i, signature in enumerate(signature_order)})
    source = source.merge(transfer_metric, on="peptide_group", how="left")
    source_path = TABLES / "tcell_signature_effect_ci_vs_no_interaction_source.csv"
    source.to_csv(source_path, index=False)

    color_col = "geomean_dominant_peptide_umi_per_10k_dc_dcbc_umi"
    values = source[color_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    color_max = float(np.nanmax(values)) if len(values) else 1.0
    if not np.isfinite(color_max) or color_max <= 0:
        color_max = 1.0
    cmap = LinearSegmentedColormap.from_list(
        "dominant_peptide_transfer_red",
        ["#f7fbff", "#f9d6d6", "#ed8590", "#c83f58"],
    )
    norm = Normalize(vmin=0.0, vmax=color_max)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for ax, signature in zip(axes.ravel(), signature_order):
        sub = source[source["signature"].eq(signature)].sort_values("peptide_order")
        x = sub["peptide_order"].to_numpy(float)
        y = sub["median_diff"].to_numpy(float)
        if len(sub):
            yerr = np.vstack([y - sub["median_diff_ci95_low"].to_numpy(float), sub["median_diff_ci95_high"].to_numpy(float) - y])
        else:
            yerr = np.empty((2, 0))
        sub_metric = sub[color_col].replace([np.inf, -np.inf], np.nan)
        has_metric = sub_metric.notna().to_numpy()
        ax.axhline(0, color="#9a9a9a", linewidth=0.45)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#555555", elinewidth=0.45, capsize=1.8, capthick=0.45)
        ax.scatter(
            x[~has_metric],
            y[~has_metric],
            s=22,
            color="#d7bde2",
            edgecolors="#333333",
            linewidths=0.25,
            zorder=3,
        )
        ax.scatter(
            x[has_metric],
            y[has_metric],
            s=22,
            c=sub_metric.loc[has_metric].to_numpy(dtype=float),
            cmap=cmap,
            norm=norm,
            edgecolors="#333333",
            linewidths=0.25,
            zorder=3,
        )
        sig = sub["q_value_bh_by_signature"].to_numpy(float) < 0.05
        ax.scatter(x[sig], y[sig], s=44, facecolors="none", edgecolors="#333333", linewidths=0.45, zorder=4)
        ax.set_title(signature, fontsize=8)
        ax.set_ylabel("Median score difference")
        ax.grid(axis="y", color="#eeeeee", linewidth=0.35)
    for ax in axes[-1, :]:
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(["multi" if group == "multi_peptide" else group for group in group_order], rotation=45, ha="right", fontsize=5.5)
    fig.suptitle("Program score shifts vs OTI no-interaction", fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.018)
    cbar.set_label("Geomean peptide UMI per 10k DC UMI", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    files = save_dual(fig, "tcell_signature_effect_ci_vs_no_interaction")
    manifest.append({"figure": "T cell signature effect CI vs no-interaction", "source_table": str(source_path), **files, "script_function": "plot_tcell_signature_effect_ci_vs_no_interaction_ordered"})
    return source


def run_tcell_peptide_dge_for_highlighted(
    features: pd.DataFrame,
    barcodes: list[str],
    matrix: sparse.csc_matrix,
    t_meta: pd.DataFrame,
    peptide_order: list[str],
    comparison_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = derive_oti_peptide_groups(t_meta, peptide_order, comparison_order + ["no_interaction"], group_col="dge_group")
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    cells["matrix_col"] = cells["CellBC"].map(barcode_to_idx)
    cells = cells.dropna(subset=["matrix_col"]).copy()
    cells["matrix_col"] = cells["matrix_col"].astype(int)
    cells = cells.sort_values(["dge_group", "CellBC"]).reset_index(drop=True)

    raw = matrix[:, cells["matrix_col"].to_numpy()].tocsc()
    features_unique, raw = unique_gene_expression(features, raw)
    genes = features_unique["gene_name"].astype(str).to_numpy()
    lib_size = np.asarray(raw.sum(axis=0)).ravel().astype(float)
    norm = raw @ sparse.diags(10000.0 / np.maximum(lib_size, 1.0))
    log_norm = norm.copy()
    log_norm.data = np.log1p(log_norm.data)
    cells["gex_umi"] = lib_size

    ref_idx = np.where(cells["dge_group"].astype(str).to_numpy() == "no_interaction")[0]
    pseudocount = 0.1
    min_total_counts = 10
    min_pct = 0.05
    results = []
    for group in comparison_order:
        idx = np.where(cells["dge_group"].astype(str).to_numpy() == group)[0]
        if len(idx) < 10 or len(ref_idx) < 10:
            continue
        pct_g = np.asarray((raw[:, idx] > 0).mean(axis=1)).ravel()
        pct_ref = np.asarray((raw[:, ref_idx] > 0).mean(axis=1)).ravel()
        mean_norm_g = np.asarray(norm[:, idx].mean(axis=1)).ravel()
        mean_norm_ref = np.asarray(norm[:, ref_idx].mean(axis=1)).ravel()
        mean_log_g, var_log_g = sparse_row_mean_var(log_norm, idx)
        mean_log_ref, var_log_ref = sparse_row_mean_var(log_norm, ref_idx)
        total_counts = np.asarray(raw[:, np.r_[idx, ref_idx]].sum(axis=1)).ravel()
        tested = (total_counts >= min_total_counts) & ((pct_g >= min_pct) | (pct_ref >= min_pct))
        se = np.sqrt(var_log_g / max(len(idx), 1) + var_log_ref / max(len(ref_idx), 1))
        t_stat = np.divide(mean_log_g - mean_log_ref, se, out=np.zeros_like(se), where=se > 0)
        denom = ((var_log_g / max(len(idx), 1)) ** 2 / max(len(idx) - 1, 1)) + ((var_log_ref / max(len(ref_idx), 1)) ** 2 / max(len(ref_idx) - 1, 1))
        df = np.divide((var_log_g / max(len(idx), 1) + var_log_ref / max(len(ref_idx), 1)) ** 2, denom, out=np.full_like(denom, np.nan), where=denom > 0)
        p_value = 2 * stats.t.sf(np.abs(t_stat), df)
        p_value[~np.isfinite(p_value)] = 1.0
        padj = np.full_like(p_value, np.nan)
        padj[tested] = bh_adjust(p_value[tested])
        pooled = np.sqrt(((len(idx) - 1) * var_log_g + (len(ref_idx) - 1) * var_log_ref) / max(len(idx) + len(ref_idx) - 2, 1))
        cohen = np.divide(mean_log_g - mean_log_ref, pooled, out=np.zeros_like(pooled), where=pooled > 0)
        detection_log2_or = np.log2(((pct_g + 0.01) / (1.01 - pct_g)) / ((pct_ref + 0.01) / (1.01 - pct_ref)))
        results.append(
            pd.DataFrame(
                {
                    "comparison": f"{group}_vs_no_interaction",
                    "peptide_group": group,
                    "reference": "no_interaction",
                    "gene": genes,
                    "total_counts": total_counts,
                    "pct_group": pct_g,
                    "pct_reference": pct_ref,
                    "pct_delta": pct_g - pct_ref,
                    "detection_log2_odds_ratio": detection_log2_or,
                    "mean_norm_group": mean_norm_g,
                    "mean_norm_reference": mean_norm_ref,
                    "mean_log1p_group": mean_log_g,
                    "mean_log1p_reference": mean_log_ref,
                    "avg_log1p_diff": mean_log_g - mean_log_ref,
                    "log2fc_mean_norm": np.log2((mean_norm_g + pseudocount) / (mean_norm_ref + pseudocount)),
                    "welch_t": t_stat,
                    "welch_df": df,
                    "cohen_d_log1p": cohen,
                    "p_value": p_value,
                    "tested": tested,
                    "padj_bh": padj,
                    "n_group_cells": len(idx),
                    "n_reference_cells": len(ref_idx),
                }
            )
        )
    dge = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not dge.empty:
        dge["is_sig_0p25"] = dge["tested"] & (dge["padj_bh"] < 0.05) & (dge["log2fc_mean_norm"].abs() >= 0.25)
        dge["is_sig_0p5"] = dge["tested"] & (dge["padj_bh"] < 0.05) & (dge["log2fc_mean_norm"].abs() >= 0.5)
    return dge, cells


def highlighted_gene_bubble_source_from_dge(dge: pd.DataFrame, group_order: list[str]) -> pd.DataFrame:
    gene_to_program = {gene: program for program, genes in HIGHLIGHTED_GENE_GROUPS.items() for gene in genes}
    genes = [gene for gene in HIGHLIGHTED_GENES if gene in set(dge["gene"])]
    source = dge[dge["gene"].isin(genes) & dge["peptide_group"].isin(group_order)].copy()
    source["gene_program"] = source["gene"].map(gene_to_program)
    source["gene_order"] = source["gene"].map({gene: i for i, gene in enumerate(genes)})
    source["peptide_order"] = source["peptide_group"].map({group: i for i, group in enumerate(group_order)})
    source["is_sig_abs0p5_fdr0p05"] = source["tested"] & (source["padj_bh"] < 0.05) & (source["log2fc_mean_norm"].abs() >= 0.5)
    source["plot_size"] = 8 + source["pct_group"].to_numpy(float) * 75
    source["x"] = source["peptide_order"]
    source["y"] = len(genes) - 1 - source["gene_order"]
    return source.sort_values(["gene_order", "peptide_order"])


def plot_tcell_highlighted_gene_relative_pattern_bubble_ordered(
    highlighted_gene_source: pd.DataFrame,
    group_order: list[str],
    manifest: list[dict[str, str]],
) -> pd.DataFrame:
    source = highlighted_gene_source[highlighted_gene_source["peptide_group"].isin(group_order)].copy()
    genes = source.sort_values("gene_order")["gene"].drop_duplicates().tolist()
    pivot = source.pivot(index="gene", columns="peptide_group", values="log2fc_mean_norm").reindex(index=genes, columns=group_order)
    z = pivot.sub(pivot.mean(axis=1), axis=0).div(pivot.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    z_long = z.reset_index().melt(id_vars="gene", var_name="peptide_group", value_name="gene_centered_log2fc_z")
    bubble = source.merge(z_long, on=["gene", "peptide_group"], how="left")
    bubble["x"] = bubble["peptide_group"].map({group: i for i, group in enumerate(group_order)})
    bubble["y"] = len(genes) - 1 - bubble["gene_order"]
    bubble["percent_detected"] = bubble["pct_group"] * 100
    bubble["plot_size"] = 5 + bubble["pct_group"].clip(lower=0, upper=1) * 72
    source_path = TABLES / "tcell_highlighted_gene_relative_pattern_bubble_source.csv"
    bubble.to_csv(source_path, index=False)

    vmax = 2.2
    cmap = LinearSegmentedColormap.from_list("relative_gene_bubble", [HEATMAP_LOW, HEATMAP_MID, HEATMAP_HIGH])
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    fig, ax = plt.subplots(figsize=(6.9, max(4.4, 0.16 * len(genes))))
    nonsig = ~bubble["is_sig_abs0p5_fdr0p05"].fillna(False)
    ax.scatter(
        bubble.loc[nonsig, "x"],
        bubble.loc[nonsig, "y"],
        s=bubble.loc[nonsig, "plot_size"],
        c=bubble.loc[nonsig, "gene_centered_log2fc_z"],
        cmap=cmap,
        norm=norm,
        edgecolors="#c0c0c0",
        linewidths=0.08,
        alpha=0.9,
    )
    sc = ax.scatter(
        bubble.loc[~nonsig, "x"],
        bubble.loc[~nonsig, "y"],
        s=bubble.loc[~nonsig, "plot_size"],
        c=bubble.loc[~nonsig, "gene_centered_log2fc_z"],
        cmap=cmap,
        norm=norm,
        edgecolors="#333333",
        linewidths=0.26,
        alpha=0.96,
    )
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(["multi" if group == "multi_peptide" else group for group in group_order], rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(list(reversed(genes)), fontsize=5.5)
    ax.set_xlabel("Peptide group")
    ax.set_ylabel("Highlighted gene")
    ax.set_title("Relative peptide pattern of highlighted genes", fontsize=8)
    ax.set_xlim(-0.55, len(group_order) - 0.45)
    ax.set_ylim(-0.7, len(genes) - 0.3)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Gene-centered log2FC z-score", fontsize=6)
    for pct in [25, 50, 75]:
        ax.scatter([], [], s=5 + (pct / 100) * 72, color="#eeeeee", edgecolors="#555555", linewidths=0.15, label=f"{pct}%")
    ax.legend(title="Detected", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=5.3, title_fontsize=5.5, borderaxespad=0)
    files = save_dual(fig, "tcell_highlighted_gene_relative_pattern_bubble")
    manifest.append({"figure": "T cell highlighted gene relative pattern bubble", "source_table": str(source_path), **files, "script_function": "plot_tcell_highlighted_gene_relative_pattern_bubble_ordered"})
    return bubble


def generate_tcell_signature_figures(
    config: dict[str, Any],
    inputs: dict[str, Any],
    features: pd.DataFrame,
    barcodes: list[str],
    matrix: sparse.csc_matrix,
    edges: pd.DataFrame,
    peptide_order: list[str],
    manifest: list[dict[str, str]],
) -> dict[str, Any]:
    full_order, noncontrol_order, order_table, lfc_metric = single_cell_lfc_peptide_order(peptide_order)
    orders = signature_group_orders(noncontrol_order)
    order_table.to_csv(TABLES / "t_cell_signature_peptide_order_single_cell_lfc.csv", index=False)

    scores, signature_genes = compute_t_cell_signature_scores(features, barcodes, matrix, inputs["t_meta"], full_order, orders["signature_heatmap"])
    scores.to_csv(TABLES / "t_cell_signature_cell_scores.csv.gz", index=False)
    signature_genes.to_csv(TABLES / "t_cell_signature_gene_sets.csv", index=False)
    group_summary = summarize_t_cell_signature_groups(scores, orders["signature_heatmap"])
    group_summary.to_csv(TABLES / "t_cell_signature_group_summary.csv", index=False)
    pairwise = t_cell_signature_pairwise_vs_no_interaction(scores, orders["signature_heatmap"])
    pairwise.to_csv(TABLES / "t_cell_signature_pairwise_vs_no_interaction.csv", index=False)

    heatmap_z = plot_t_cell_signature_group_median_heatmap_ordered(group_summary, list(T_CELL_SIGNATURES), orders["signature_heatmap"], manifest)
    transfer_metric = tcell_signature_effect_transfer_color_metric(edges, inputs["t_meta"], orders["effect_ci"])
    effect_source = plot_tcell_signature_effect_ci_vs_no_interaction_ordered(
        pairwise,
        KEY_T_CELL_SIGNATURES,
        orders["effect_ci"],
        manifest,
        transfer_metric,
    )

    dge, dge_cells = run_tcell_peptide_dge_for_highlighted(features, barcodes, matrix, inputs["t_meta"], full_order, orders["highlighted_gene_bubble"])
    dge_cells[
        ["CellBC", "dge_group", "total_barcode_umi", "assigned_dcbc_umi", "interaction_state", "AssignedPeptideBC_Name", "AssignedTreatment", "gex_umi"]
    ].to_csv(TABLES / "tcell_peptide_dge_cells_used.csv", index=False)
    dge.to_csv(TABLES / "tcell_peptide_dge_vs_no_interaction.csv.gz", index=False)
    if not dge.empty:
        dge_summary = (
            dge.groupby("peptide_group", as_index=False)
            .agg(
                n_tested_genes=("tested", "sum"),
                n_sig_abs0p5_fdr0p05=("is_sig_0p5", "sum"),
                n_group_cells=("n_group_cells", "first"),
                n_reference_cells=("n_reference_cells", "first"),
            )
            .sort_values("peptide_group", key=lambda s: s.map({g: i for i, g in enumerate(orders["highlighted_gene_bubble"])}))
        )
    else:
        dge_summary = pd.DataFrame()
    dge_summary.to_csv(TABLES / "tcell_peptide_dge_summary_by_comparison.csv", index=False)
    highlighted_source = highlighted_gene_bubble_source_from_dge(dge, orders["highlighted_gene_bubble"]) if not dge.empty else pd.DataFrame()
    highlighted_source.to_csv(TABLES / "tcell_peptide_dge_highlighted_gene_bubble_source.csv", index=False)
    relative_bubble = plot_tcell_highlighted_gene_relative_pattern_bubble_ordered(highlighted_source, orders["highlighted_gene_bubble"], manifest)

    summary = {
        "n_oti_cells_scored": int(len(scores)),
        "peptide_group_counts": {str(k): int(v) for k, v in scores["peptide_group"].astype(str).value_counts().reindex(orders["signature_heatmap"]).fillna(0).to_dict().items()},
        "n_signatures": int(len(T_CELL_SIGNATURES)),
        "single_cell_lfc_order_metric": lfc_metric,
        "noncontrol_peptide_order_by_single_cell_lfc_desc": noncontrol_order,
        "signature_heatmap_group_order": orders["signature_heatmap"],
        "signature_effect_group_order": orders["effect_ci"],
        "signature_effect_color_metric": "geomean peptide-assigned DCBC UMI among qualifying OT-I cells per 10,000 dendritic-cell supported peptide DCBC UMI",
        "signature_effect_color_metric_table": str(TABLES / "tcell_signature_effect_ci_peptide_transfer_color_metric.csv"),
        "signature_effect_qualifying_cells_table": str(TABLES / "tcell_signature_effect_ci_peptide_transfer_qualifying_cells.csv"),
        "signature_effect_cell_peptide_umi_min_exclusive": PEPTIDE_TRANSFER_UMI_MIN_EXCLUSIVE,
        "signature_effect_cell_peptide_fraction_min": PEPTIDE_TRANSFER_FRACTION_MIN,
        "signature_effect_color_group_without_metric": "multi_peptide",
        "highlighted_gene_bubble_group_order": orders["highlighted_gene_bubble"],
        "n_effect_ci_rows": int(len(effect_source)),
        "n_highlighted_relative_bubble_rows": int(len(relative_bubble)),
        "n_tcell_dge_rows": int(len(dge)),
        "module_score_method": "Mean of per-gene z-scored log1p(CP10k) expression across OTI cells.",
    }
    write_json(summary, QC / "t_cell_signature_figure_updates_summary.json")
    return summary


def heatmap_groups_and_plots(config: dict[str, Any], inputs: dict[str, Any], edges: pd.DataFrame, manifest: list[dict[str, str]]) -> dict[str, Any]:
    features, barcodes, matrix = load_10x_matrix(config)
    gene_to_indices = build_gene_index(features)
    dc_meta = inputs["dc_meta"].copy()
    t_meta = inputs["t_meta"].copy()
    treatment_order = config["orders"]["treatments"]
    peptide_order = config["orders"]["peptides"]

    dc_assign = dc_meta[(pd.to_numeric(dc_meta["top_DCBC_fraction"], errors="coerce") > 0.90) & dc_meta["AssignedTreatment"].astype(str).ne("")]
    dc_groups = {t: dc_assign.loc[dc_assign["AssignedTreatment"] == t, "CellBC"].astype(str).tolist() for t in treatment_order}
    dc_mean, dc_positive, dc_missing = expression_by_group(matrix, gene_to_indices, barcodes, config["dc_gene_panel"], dc_groups)
    dc_mean, dc_positive, dc_blank = drop_blank_expression_rows(dc_mean, dc_positive, treatment_order)
    dc_z = zscore_rows(dc_mean, treatment_order)
    dc_mean.to_csv(TABLES / "dc_phenotype_heatmap_mean_expression.csv", index=False)
    dc_z.to_csv(TABLES / "dc_phenotype_heatmap_zscore.csv", index=False)
    clustered_heatmap(dc_z, treatment_order, "DC phenotype heatmap", "dc_phenotype_heatmap_clustered", manifest, TABLES / "dc_phenotype_heatmap_zscore.csv")

    t_groups = {
        "no_treatment": t_meta.loc[(t_meta["t_cell_type"] == "OTI") & (t_meta["AssignedTreatment"] == "no_treatment"), "CellBC"].astype(str).tolist(),
        "LPS": t_meta.loc[(t_meta["t_cell_type"] == "OTI") & (t_meta["AssignedTreatment"] == "LPS"), "CellBC"].astype(str).tolist(),
        "PolyIC": t_meta.loc[(t_meta["t_cell_type"] == "OTI") & (t_meta["AssignedTreatment"] == "PolyIC"), "CellBC"].astype(str).tolist(),
        "IFNg": t_meta.loc[(t_meta["t_cell_type"] == "OTI") & (t_meta["AssignedTreatment"] == "IFNg"), "CellBC"].astype(str).tolist(),
        "Multiple": t_meta.loc[(t_meta["t_cell_type"] == "OTI") & (t_meta["interaction_state"] == "multi_interaction"), "CellBC"].astype(str).tolist(),
        "OTI no interaction": t_meta.loc[(t_meta["t_cell_type"] == "OTI") & (t_meta["interaction_state"] == "no_interaction"), "CellBC"].astype(str).tolist(),
        "C57BL/6": t_meta.loc[t_meta["t_cell_type"] == "C57BL6", "CellBC"].astype(str).tolist(),
    }
    t_treatment_cols = list(t_groups)
    t_mean, t_positive, t_missing = expression_by_group(matrix, gene_to_indices, barcodes, config["t_cell_gene_panel"], t_groups)
    t_mean, t_positive, t_blank = drop_blank_expression_rows(t_mean, t_positive, t_treatment_cols)
    t_z = zscore_rows(t_mean, t_treatment_cols)
    t_mean.to_csv(TABLES / "t_cell_treatment_heatmap_mean_expression.csv", index=False)
    t_positive.to_csv(TABLES / "t_cell_treatment_heatmap_positive_mean_expression.csv", index=False)
    t_z.to_csv(TABLES / "t_cell_treatment_heatmap_zscore.csv", index=False)
    clustered_heatmap(t_z, t_treatment_cols, "T cell phenotype heatmap by treatment", "t_cell_treatment_heatmap_clustered", manifest, TABLES / "t_cell_treatment_heatmap_zscore.csv")

    oti_meta = t_meta[t_meta["t_cell_type"] == "OTI"].copy()
    peptide_groups = {p: oti_meta.loc[oti_meta["AssignedPeptideBC_Name"] == p, "CellBC"].astype(str).tolist() for p in peptide_order}
    peptide_groups["multi_peptide"] = oti_meta.loc[
        oti_meta["AssignedPeptideBC_Name"].astype(str).eq("") & (oti_meta["interaction_state"] != "no_interaction"),
        "CellBC",
    ].astype(str).tolist()
    peptide_groups["no_interaction"] = oti_meta.loc[oti_meta["interaction_state"] == "no_interaction", "CellBC"].astype(str).tolist()
    peptide_cols = peptide_order + ["multi_peptide", "no_interaction"]
    pep_mean, _pep_positive, pep_missing = expression_by_group(matrix, gene_to_indices, barcodes, config["t_cell_gene_panel"], peptide_groups)
    pep_mean, _pep_positive, pep_blank = drop_blank_expression_rows(pep_mean, _pep_positive, peptide_cols)
    pep_z = zscore_rows(pep_mean, peptide_cols)
    pep_mean.to_csv(TABLES / "t_cell_peptide_heatmap_mean_expression.csv", index=False)
    pep_z.to_csv(TABLES / "t_cell_peptide_heatmap_zscore.csv", index=False)
    clustered_heatmap(pep_z, peptide_cols, "OTI T cell phenotype heatmap by peptide", "t_cell_peptide_heatmap_clustered", manifest, TABLES / "t_cell_peptide_heatmap_zscore.csv")

    gene_sets = {
        "No-treatment": ["Fos", "Fosb", "Dusp1", "Rela", "Cd27", "Prf1", "S1pr5", "Cxcr6"],
        "LPS": ["Rorc", "Csf2", "Il21", "Gzmm", "Gzmc", "Cx3cr1", "Cd38", "Nfkbia"],
        "PolyIC": ["Il17a", "Ccr6", "Ccr2", "Lair1"],
        "IFNg": ["Cd40lg", "Cd226", "Il10", "Entpd1", "Itgam"],
    }
    plot_gene_set_bubble(t_z, t_positive, t_treatment_cols, gene_sets, manifest)
    tcell_signature_summary = generate_tcell_signature_figures(config, inputs, features, barcodes, matrix, edges, peptide_order, manifest)
    return {
        "dc_missing_genes": dc_missing,
        "dc_blank_genes_removed": dc_blank,
        "t_treatment_missing_genes": t_missing,
        "t_treatment_blank_genes_removed": t_blank,
        "t_peptide_missing_genes": pep_missing,
        "t_peptide_blank_genes_removed": pep_blank,
        "t_peptide_group_cell_counts": {k: len(v) for k, v in peptide_groups.items()},
        "treatment_group_cell_counts": {k: len(v) for k, v in t_groups.items()},
        "dc_group_cell_counts": {k: len(v) for k, v in dc_groups.items()},
        "tcell_signature_figure_updates": tcell_signature_summary,
    }


def plot_gene_set_bubble(z_df: pd.DataFrame, positive_df: pd.DataFrame, cols: list[str], gene_sets: dict[str, list[str]], manifest: list[dict[str, str]]) -> None:
    z_map = z_df.set_index("gene")
    pos_map = positive_df.set_index("gene")
    rows = []
    y_labels = []
    y = 0
    group_boundaries = []
    for group, genes in gene_sets.items():
        start = y
        for gene in genes:
            y_labels.append(gene)
            for x, col in enumerate(cols):
                rows.append({"gene_set": group, "gene": gene, "group": col, "x": x, "y": y, "zscore": float(z_map.loc[gene, col]) if gene in z_map.index else np.nan, "positive_mean_expression": float(pos_map.loc[gene, col]) if gene in pos_map.index else 0.0})
            y += 1
        group_boundaries.append((group, start, y - 1))
        y += 1
    df = pd.DataFrame(rows)
    x_spacing = 1.12
    y_spacing = 1.46
    df["x_plot"] = df["x"] * x_spacing
    df["y_plot"] = df["y"] * y_spacing
    out_table = TABLES / "t_cell_select_gene_set_bubble_source.csv"
    df.to_csv(out_table, index=False)
    fig, ax = plt.subplots(figsize=(4.45, 6.8))
    plot_df = df.dropna(subset=["zscore"]).copy()
    size_max = max(float(plot_df["positive_mean_expression"].max()), 1e-12)
    def size_for(values: np.ndarray | float) -> np.ndarray | float:
        return 12 + (np.asarray(values, dtype=float) / size_max) * 145

    sizes = size_for(plot_df["positive_mean_expression"].to_numpy(float))
    cmap = LinearSegmentedColormap.from_list("zdiv", ["#5d8af7", "#ffffff", "#ed8590"])
    ax.scatter(plot_df["x_plot"], plot_df["y_plot"], s=sizes, c=plot_df["zscore"], cmap=cmap, vmin=-2, vmax=2, edgecolors="#555555", linewidths=0.25)
    ax.set_xticks([i * x_spacing for i in range(len(cols))])
    ax.set_xticklabels(cols, rotation=45, ha="right")
    valid_y = sorted(df["y_plot"].unique())
    ax.set_yticks(valid_y)
    ax.set_yticklabels([])
    # Draw explicit gene labels as text to allow group gaps.
    gene_at_y = df.drop_duplicates("y")[["gene", "y_plot"]]
    for row in gene_at_y.itertuples(index=False):
        ax.text(-0.72, row.y_plot, row.gene, fontsize=5.5, ha="right", va="center")
    max_x = (len(cols) - 1) * x_spacing
    for group, start, end in group_boundaries:
        ax.text(max_x + 0.55, ((start + end) / 2) * y_spacing, group, fontsize=6, va="center", ha="left")
        ax.axhline((end + 0.5) * y_spacing, color="#dddddd", linewidth=0.5)
    ax.set_xlim(-0.86, max_x + 1.95)
    ax.set_ylim(max(valid_y) + 0.9, -0.9)
    ax.set_title("Select T cell phenotype genes")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-2, vmax=2))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Z-score", fontsize=6)
    positive_values = plot_df.loc[plot_df["positive_mean_expression"] > 0, "positive_mean_expression"].to_numpy(float)
    if len(positive_values):
        legend_values = np.unique(np.round(np.percentile(positive_values, [25, 50, 90]), 2))
        handles = [
            ax.scatter([], [], s=float(size_for(value)), facecolor="#cfcfcf", edgecolor="#555555", linewidth=0.25)
            for value in legend_values
        ]
        ax.legend(
            handles,
            [f"{value:g}" for value in legend_values],
            title="Mean expr+ cells",
            fontsize=5,
            title_fontsize=5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=len(legend_values),
            columnspacing=1.2,
            handletextpad=0.8,
        )
    files = save_dual(fig, "t_cell_select_gene_set_bubble")
    manifest.append({"figure": "T cell select gene set bubble", "source_table": str(out_table), **files})


def violin(ax: plt.Axes, data: list[np.ndarray], positions: list[float], colors: list[str], widths: float = 0.65) -> None:
    parts = ax.violinplot(data, positions=positions, widths=widths, showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#555555")
        body.set_linewidth(0.4)
        body.set_alpha(0.95)
    for arr, pos in zip(data, positions):
        if len(arr):
            q1, median, q3 = np.percentile(arr, [25, 50, 75])
            ax.plot([pos, pos], [q1, q3], color="#333333", linewidth=0.75, zorder=3)
            ax.scatter([pos], [median], s=14, marker="o", facecolor="#ffffff", edgecolor="#333333", linewidth=0.55, zorder=4)


def umi_violin_plots(inputs: dict[str, Any], manifest: list[dict[str, str]]) -> None:
    dc_meta = inputs["dc_meta"].copy()
    t_meta = inputs["t_meta"].copy()
    dc = dc_meta[(pd.to_numeric(dc_meta["top_DCBC_fraction"], errors="coerce") > 0.95) & dc_meta["AssignedTreatment"].astype(str).ne("")].copy()
    dc.to_csv(TABLES / "dc_umi_per_cell_violin_source.csv", index=False)
    treatment_order = ["no_treatment", "LPS", "PolyIC", "IFNg"]
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    data = [np.log10(dc.loc[dc["AssignedTreatment"] == t, "total_barcode_umi"].astype(float) + 1) for t in treatment_order]
    violin(ax, data, list(range(len(treatment_order))), [C57_GREY for _ in treatment_order])
    ax.set_xticks(range(len(treatment_order)))
    ax.set_xticklabels(treatment_order, rotation=35, ha="right")
    ax.set_ylabel("log10(DCBC UMI + 1)")
    ax.set_title("DC UMI per cell")
    files = save_dual(fig, "dc_umi_per_cell_by_treatment_violin")
    manifest.append({"figure": "UMI per cell DC", "source_table": str(TABLES / "dc_umi_per_cell_violin_source.csv"), **files})

    t_meta.to_csv(TABLES / "t_cell_umi_per_cell_violin_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(1.9, 2.1))
    groups = ["OTI", "C57BL6"]
    data = [np.log10(t_meta.loc[t_meta["t_cell_type"] == g, "total_barcode_umi"].astype(float) + 1) for g in groups]
    violin(ax, data, [0, 1], [OTI_BLUE, C57_GREY], widths=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["OTI", "C57BL/6"])
    ax.set_ylabel("log10(DCBC UMI + 1)")
    ax.set_title("T cell UMI per cell")
    files = save_dual(fig, "t_cell_umi_per_cell_violin")
    manifest.append({"figure": "UMI per cell OTI and C57BL6", "source_table": str(TABLES / "t_cell_umi_per_cell_violin_source.csv"), **files})


def interaction_fingerprints(inputs: dict[str, Any], edges: pd.DataFrame, metrics: pd.DataFrame, manifest: list[dict[str, str]], rng: np.random.Generator, peptide_order: list[str]) -> dict[str, Any]:
    t_edges = edges[(edges["cell_class"] == "t_cell") & (edges["assigned_dcbc"])].copy()
    all_samples = []
    sample_summary = {}
    for t_type in ["OTI", "C57BL6"]:
        sub = metrics[(metrics["t_cell_type"] == t_type) & (metrics["unique_dcbc"] > 0)].copy()
        if sub.empty:
            continue
        selected_cells: set[str] = set()
        pools = {
            "high": sub[sub["unique_dcbc"] >= sub["unique_dcbc"].quantile(0.75)],
            "low": sub[sub["unique_dcbc"] <= sub["unique_dcbc"].quantile(0.25)],
        }
        for label in ["high", "low"]:
            pool = pools[label]
            if label == "low":
                pool = pool[~pool["CellBC"].isin(selected_cells)]
            n = min(10, len(pool))
            sampled = pool.sample(n=n, random_state=int(rng.integers(1, 10_000_000))) if n else pool
            selected_cells.update(sampled["CellBC"].astype(str))
            sampled = sampled.assign(t_cell_type=t_type, fingerprint_level=label, sample_group=f"{t_type}_{label}")
            all_samples.append(sampled)
            sample_summary[f"{t_type}_{label}"] = int(n)
    if not all_samples:
        return sample_summary
    samples = pd.concat(all_samples, ignore_index=True)
    rows = []
    for sampled_cell in samples.itertuples(index=False):
        cell = sampled_cell.CellBC
        sub = t_edges[t_edges["CellBC"] == cell]
        total = float(sub["UMI"].sum())
        by_peptide = sub.groupby("AssignedPeptideBC_Name")["UMI"].sum()
        for peptide in peptide_order:
            umi = int(by_peptide.get(peptide, 0))
            rows.append(
                {
                    "CellBC": cell,
                    "t_cell_type": sampled_cell.t_cell_type,
                    "fingerprint_level": sampled_cell.fingerprint_level,
                    "sample_group": sampled_cell.sample_group,
                    "PeptideBC_Name": peptide,
                    "UMI": umi,
                    "proportion": float(umi / total) if total else 0.0,
                }
            )
    source = pd.DataFrame(rows)
    out_table = TABLES / "single_cell_interaction_fingerprints_source.csv"
    source.to_csv(out_table, index=False)
    for t_type, group_cells in samples.groupby("t_cell_type", sort=False):
        group_cells = group_cells.copy()
        group_cells["level_order"] = group_cells["fingerprint_level"].map({"high": 0, "low": 1})
        group_cells = group_cells.sort_values(["level_order", "unique_dcbc", "total_dcbc_umi"], ascending=[True, False, False])
        plot_df = source[source["CellBC"].isin(group_cells["CellBC"])].copy()
        cells = group_cells["CellBC"].tolist()
        fig, ax = plt.subplots(figsize=(4.7, max(3.0, 0.15 * len(cells) + 1.15)))
        left = np.zeros(len(cells))
        y = np.arange(len(cells))
        for peptide in peptide_order:
            vals = np.array([plot_df.loc[(plot_df["CellBC"] == cell) & (plot_df["PeptideBC_Name"] == peptide), "proportion"].sum() for cell in cells])
            ax.barh(y, vals, left=left, color=PEPTIDE_COLORS[peptide], linewidth=0, label=peptide)
            left += vals
        ax.set_yticks(y)
        label_counts = {"high": 0, "low": 0}
        y_labels = []
        for level in group_cells["fingerprint_level"]:
            label_counts[level] += 1
            y_labels.append(f"{level[0].upper()}{label_counts[level]}")
        ax.set_yticklabels(y_labels, fontsize=6)
        high_n = int((group_cells["fingerprint_level"] == "high").sum())
        if 0 < high_n < len(group_cells):
            ax.axhline(high_n - 0.5, color="#555555", linewidth=0.55)
            ax.text(1.01, (high_n - 1) / 2, "High", transform=ax.get_yaxis_transform(), fontsize=6, va="center", ha="left")
            ax.text(1.01, high_n + (len(group_cells) - high_n - 1) / 2, "Low", transform=ax.get_yaxis_transform(), fontsize=6, va="center", ha="left")
        ax.invert_yaxis()
        ax.set_xlabel("Peptide fraction of DCBC UMI")
        ax.set_ylabel("Sampled cell")
        ax.set_title(f"{t_type} interaction fingerprints")
        ax.legend(ncol=4, fontsize=4.8, loc="lower center", bbox_to_anchor=(0.5, -0.48))
        files = save_dual(fig, f"interaction_fingerprint_{t_type}")
        manifest.append({"figure": f"Single cell interaction fingerprints {t_type}", "source_table": str(out_table), **files})
    return sample_summary


def interaction_strength_distribution(metrics: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    out_table = TABLES / "interaction_strength_distribution_source.csv"
    metrics.to_csv(out_table, index=False)
    positive = metrics.loc[metrics["total_dcbc_umi"] > 0, "total_dcbc_umi"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    curve_rows = []
    if len(positive):
        bins = np.unique(np.logspace(0, np.log10(max(positive.max(), 1)), 45).astype(int))
        if len(bins) < 2:
            bins = np.array([1, int(positive.max()) + 1])
        centers = np.sqrt(bins[:-1] * bins[1:])
        for t_type, color, label in [("OTI", OTI_BLUE, "OTI"), ("C57BL6", C57_GREY, "C57BL/6")]:
            vals = metrics.loc[(metrics["t_cell_type"] == t_type) & (metrics["total_dcbc_umi"] > 0), "total_dcbc_umi"].to_numpy(dtype=float)
            counts, _edges = np.histogram(vals, bins=bins)
            zero_count = int(((metrics["t_cell_type"] == t_type) & (metrics["total_dcbc_umi"] == 0)).sum())
            for left, right, center, count in zip(bins[:-1], bins[1:], centers, counts):
                curve_rows.append({"t_cell_type": t_type, "bin_left": int(left), "bin_right": int(right), "bin_center": float(center), "n_t_cells": int(count), "zero_umi_cells": zero_count})
            ax.plot(centers, counts, color=color, linewidth=1.3, drawstyle="steps-mid", label=label)
        ax.set_xscale("log")
    pd.DataFrame(curve_rows).to_csv(TABLES / "interaction_strength_distribution_curve_source.csv", index=False)
    ax.set_xlabel("Total DCBC UMI per T cell")
    ax.set_ylabel("Number of T cells")
    ax.set_title("Interaction strength")
    ax.legend(fontsize=6)
    files = save_dual(fig, "interaction_strength_distribution")
    manifest.append({"figure": "Interaction strength per T cell", "source_table": str(out_table), **files})


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(475)
    config = load_config()
    inputs = load_inputs(config)
    edges = build_edge_table(inputs)
    manifest: list[dict[str, str]] = []
    summary: dict[str, Any] = {"random_seed": 475}

    peptide_order = config["orders"]["peptides"]
    treatment_order = config["orders"]["treatments"]
    summary.update(plot_standard_umap(inputs, manifest))
    summary.update(plot_interaction_umap(inputs, edges, manifest, rng))
    metrics = t_cell_dcbc_metrics(inputs, edges)
    plot_pickup_multiplicity(metrics, manifest, rng)
    bubble, _denom = condition_bubble_inputs(inputs, edges, peptide_order, treatment_order)
    plot_condition_bubbles(bubble, peptide_order, treatment_order, manifest)
    summary.update(generate_dc_supported_normalized_bubble(config, inputs, edges, peptide_order, treatment_order, manifest))
    summary.update(generate_tcell_dc_supported_normalized_boxplots(config, inputs, edges, peptide_order, treatment_order, manifest))
    summary.update(peptide_umi_lfc(edges, peptide_order, manifest))
    summary.update(run_dc_treatment_dge(config, inputs, edges, manifest))
    summary.update(heatmap_groups_and_plots(config, inputs, edges, manifest))
    umi_violin_plots(inputs, manifest)
    summary["fingerprint_sample_counts"] = interaction_fingerprints(inputs, edges, metrics, manifest, rng, peptide_order)
    interaction_strength_distribution(metrics, manifest)

    manifest_df = pd.DataFrame(manifest)
    manifest_df.insert(1, "script", str(Path(__file__).resolve()))
    manifest_df.to_csv(TABLES / "figure_manifest.csv", index=False)
    summary["n_figures"] = int(len(manifest_df))
    summary["figures"] = manifest
    summary["input_count_table"] = str(PUB / "data_intermediate" / "barcode_read_support_filtered_count_table.csv.gz")
    write_json(summary, QC / "run_summary.json")
    write_json({"n_figures": int(len(manifest_df)), "manifest": str(TABLES / "figure_manifest.csv")}, QC / "figure_generation_summary.json")


if __name__ == "__main__":
    main()
