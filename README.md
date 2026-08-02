# Relay publication code

This repository contains the analysis code and processed inputs needed to recreate publication figures generated computationally. Raw and large upstream data are distributed separately and mapped to their expected locations in `data_package_layout/`. Each workflow writes PDF and PNG figures, a run summary, a quality summary, and file hashes.

Repository URL: https://github.com/mcmanuslab/cell-relay

## Contents

All workflows use the same layout:

```text
workflows/<workflow>/
  README.md
  code/                  Figure-generation and upstream analysis code
  data/figure_tables/    Processed inputs used directly by figure replay
  data_manifest.tsv      Checksums and sizes for figure inputs
  outputs/figures/       Generated PDF, PNG, and selected SVG files
  outputs/summaries/     Generated run and quality summaries
```

| Workflow | Publication figure | Figure pairs | Extra requirement |
|---|---:|---:|---|
| `figure_1f` | Figure 1F | 2 | None |
| `figure_2i` | Figure 2I | 1 | None |
| `figure_3b` | Figure 3B | 1 | None |
| `figure_3d_f` | Figure 3D-F | 3 | MAGeCK only for full upstream rerun |
| `figure_4` | Figure 4 | 28 | None for figure replay |
| `figure_5` | Figure 5 | 34 | R and listed R packages; MAGeCK for upstream rerun |
| `extended_data_figure_4` | Extended Data Figure 4 | 8 | Poppler `pdftoppm` |
| `extended_data_figure_8` | Extended Data Figure 8 | 6 | None |

`workflow_manifest.tsv` is the machine-readable workflow index. `data_package_layout/` describes the expected locations for separately distributed raw and upstream data.

## System requirements

No non-standard hardware is required. Testing was performed on macOS 26.5.1, Apple silicon (`arm64`), with:

- Python 3.13.5
- `adjustText` 1.3.0
- `matplotlib` 3.10.6
- `numpy` 2.3.2
- `pandas` 2.3.1
- `regex` 2026.1.15
- `scipy` 1.17.0
- R 4.5.3 for the Figure 5 B-versus-C plot
- R packages: `tidyverse` 2.0.0, `ggrepel` 0.9.8, `ragg` 1.5.2, `svglite` 2.2.2
- Poppler 26.05.0 for Extended Data Figure 4 flow-panel conversion

Linux and recent macOS systems should work. Windows has not been tested; Poppler and R must be available on `PATH`. MAGeCK is not needed to recreate figures from the included tables.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the Figure 5 R dependencies once:

```bash
Rscript -e 'install.packages(c("tidyverse", "ggrepel", "ragg", "svglite"))'
```
Install Poppler if `pdftoppm -v` is unavailable. For example, use `brew install poppler` on macOS or `apt-get install poppler-utils` on Debian/Ubuntu.

Typical installation time on a normal desktop with an existing Python and R installation is 5-15 minutes, primarily dependent on network speed and whether R packages must be compiled.

## Demo

Validate the included inputs, run all figure workflows, and validate the generated outputs:

```bash
python validate_package.py
python run_all.py
python validate_package.py --require-outputs
```

The expected result is 83 PDF/PNG figure pairs across the eight workflow output directories, plus selected SVG files. Every `outputs/summaries/quality_summary.json` should report `"status": "pass"`; `validate_package.py --require-outputs` should print a JSON object with `"status": "pass"`. Generated output directories are ignored by Git and can be deleted and recreated at any time.

The tested workflows required about 90 seconds of plotting time in total after imports and font caching, and approximately 2-3 minutes wall time for a first sequential run on the test system. Figure 4 and Figure 5 are the longest individual workflows.

To run one workflow:

```bash
python run_all.py --workflow figure_4
```

The equivalent direct command is documented in each workflow README. Existing files in `outputs/` are overwritten by the corresponding figure renderer; input tables are read-only during normal figure recreation.

## Using real data

The default entry points read processed inputs from `data/figure_tables/`. To rerun an upstream analysis, obtain the separately distributed data and place its contents into the matching `workflows/<workflow>/data/raw/` and `workflows/<workflow>/data/upstream/` locations shown under `data_package_layout/`.

Full-workflow entry points and requirements are described in the workflow READMEs. In particular:

- Figure 3D-F provides the MAGeCK count/test commands in `code/run_mageck_commands.sh`.
- Figure 5 provides FASTQ-to-count-table code, MAGeCK commands, downstream screen analysis, and R plotting code.
- Figure 4 provides the upstream single-cell workflow and configuration; the external data must provide the Cell Ranger matrices, analysis archive, V(D)J annotations, barcode FASTQs, and peptide barcode reference.

`data_package_layout/omitted_data_manifest.tsv` lists externally distributed raw/upstream data categories and their target locations. Large FASTQ, BAM, FCS, Cell Ranger, archive, and database files are not stored in this repository.

## Reproducibility and quality control

Matplotlib workflows set `pdf.fonttype=42` and `ps.fonttype=42`. The Figure 5 R plot writes PDF, PNG, and SVG outputs. Each run records dependency versions, runtime, input and output SHA-256 checksums, expected-file checks, PDF/PNG pairing, and a Type 3 font scan.

After changing included figure inputs or code, run `python scripts/build_manifests.py` and then `python validate_package.py`.

## License

The software in this repository is licensed under the Apache License 2.0; see `LICENSE`.
