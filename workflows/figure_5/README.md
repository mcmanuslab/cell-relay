# Figure 5

Recreates the screen dot plot, barcode-stack plots, decontamination densities, validation heatmaps, UMI QC, and 21 program-card panels. The B-versus-C dot plot is generated with R.

```bash
python workflows/figure_5/code/recreate_figures.py
```

The default replay requires R plus `tidyverse`, `ggrepel`, `ragg`, and `svglite`. Inputs are under `data/figure_tables/`; exact checksums are in `data_manifest.tsv`. The tested output is 34 PDF/PNG pairs plus one SVG dot plot.

The full raw-data order is:

```bash
python workflows/figure_5/code/run_upstream_from_fastq.py
bash workflows/figure_5/code/run_mageck_commands.sh
python workflows/figure_5/code/workflow.py
```

Place FASTQs below `data/raw/fastq/NT466-D04/` and `data/raw/fastq/NT466-D05/`, and the library at `data/raw/reference/BC.txt`. The first command performs UMI counting, merging, Hamming-distance collapse, decontamination, and count-table construction. The shell script runs the MAGeCK tests and MLE commands using files below `data/upstream/mageck/`. Screen references, annotation cache, validation inputs, and design matrices belong under the `data/upstream/` structure documented in the external data layout.
