# Extended Data Figure 8

Recreates the A2 MFI bar plot, the flow histogram, and four RNA-seq volcano plots from included tables and one source SVG.

```bash
python workflows/extended_data_figure_8/code/recreate_figures.py
```

Inputs are under `data/figure_tables/`; exact checksums are in `data_manifest.tsv`. The tested output is six PDF/PNG pairs plus SVG versions of the histogram and volcano plots.

For a full rerun, populate `data/raw/` with FCS/FlowJo and FASTQ or BAM inputs plus the genome annotation. Place large count caches and upstream flow/RNA-seq results under `data/upstream/`, then use `code/workflow.py`. These files are not required to recreate figures from the included inputs.
