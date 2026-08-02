# Figure 2I

Recreates the UMI coupling-category stacked bar plot from the included coupling table.

```bash
python workflows/figure_2i/code/recreate_figures.py
```

Inputs are under `data/figure_tables/`; exact checksums are in `data_manifest.tsv`. Figures are written to `outputs/figures/`, and run/QC JSON files to `outputs/summaries/`.

For a full rerun, populate `data/raw/fastq/` and `data/raw/reference/`, then run `python workflows/figure_2i/code/workflow.py`. The upstream workflow parses long reads, matches barcodes/gRNAs, collapses UMIs, and builds the plotted table.
