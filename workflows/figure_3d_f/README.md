# Figure 3D-F

Recreates the OT, WT, and IFN-versus-OT panels from three included plot tables.

```bash
python workflows/figure_3d_f/code/recreate_figures.py
```

Inputs are under `data/figure_tables/`; exact checksums are in `data_manifest.tsv`. Figures are written to `outputs/figures/`, and run/QC JSON files to `outputs/summaries/`.

For a full rerun, populate `data/raw/fastq/`, `data/raw/reference/`, and `data/raw/flow_exports/`. Run the MAGeCK commands with `bash workflows/figure_3d_f/code/run_mageck_commands.sh`, then run `python workflows/figure_3d_f/code/workflow.py`. The shell script defines the sample labels, `mageck count`, and six `mageck test` calls.
