# Figure 3B

Recreates the replicate log-fold-change scatter plot from the included plot table.

```bash
python workflows/figure_3b/code/recreate_figures.py
```

Inputs are under `data/figure_tables/`; exact checksums are in `data_manifest.tsv`. Figures are written to `outputs/figures/`, and run/QC JSON files to `outputs/summaries/`.

For a full rerun, populate `data/raw/fastq/` and `data/raw/reference/` with the library and comparison definitions, then run `python workflows/figure_3b/code/workflow.py`.
