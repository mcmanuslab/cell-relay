# Figure 4

Recreates all 28 entries in `data/figure_tables/figure_manifest.csv` from the included figure-source tables.

```bash
python workflows/figure_4/code/recreate_figures.py
```

Inputs are under `data/figure_tables/`; exact checksums are in `data_manifest.tsv`. All figures use the publication styling in `code/scripts/generate_publication_figures.py` and are written to `outputs/figures/`.

For a full single-cell rerun, populate these paths:

```text
data/raw/barcode_fastq/
data/raw/reference/OTI-Peptide-BC.csv
data/raw/cellranger/sample_filtered_feature_bc_matrix.tar.gz
data/raw/cellranger/analysis.tar.gz
data/raw/cellranger/vdj_t/filtered_contig_annotations.csv
```

Then run `python workflows/figure_4/code/scripts/run_workflow.py`. The configuration is `code/config.yaml`; large intermediates are written below `data/upstream/publication_analysis/`. A full rerun can replace the included figure tables, so use a separate working copy when those inputs must be preserved.
