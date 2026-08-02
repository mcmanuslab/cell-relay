# Extended Data Figure 4

Recreates barcode-proportion panels and five flow-panel renders from included source PDFs.

```bash
python workflows/extended_data_figure_4/code/recreate_figures.py
```

Poppler `pdftoppm` must be on `PATH`. Inputs are under `data/figure_tables/`; exact checksums are in `data_manifest.tsv`. The tested output is eight PDF/PNG pairs.

For a full rerun, populate `data/raw/` with FCS, FASTQ, and barcode references and use `code/workflow.py`. Raw instrument and sequencing files are required only for the upstream analysis.
