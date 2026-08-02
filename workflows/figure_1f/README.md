# Figure 1F

Recreates the barcode log2 fold-change dot plot and pMHC violin plot from five included CSV tables.

```bash
python workflows/figure_1f/code/recreate_figures.py
```

Inputs are under `data/figure_tables/`; exact checksums are in `data_manifest.tsv`. Figures are written to `outputs/figures/`, and run/QC JSON files to `outputs/summaries/`.

For a full rerun, populate `data/raw/` with the sequencing FASTQs and references shown in `data_package_layout/`, then run `python workflows/figure_1f/code/workflow.py`. The full workflow creates count intermediates before regenerating the same figure tables.
