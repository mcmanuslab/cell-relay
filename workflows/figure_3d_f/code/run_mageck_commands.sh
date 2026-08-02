#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKDIR="$ROOT/data/upstream/mageck"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
cp "$ROOT/data/raw/reference/OTI.txt" ./

# Link raw FASTQs to the filenames recorded in the MAGeCK log.
# Mapping source: summaries/fastq_label_manifest.csv
ln -sf "$ROOT/data/raw/fastq/DC-1_S1_L001_R1_001.fastq.gz" DC-1.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-2_S2_L001_R1_001.fastq.gz" DC-2.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-IL-1_S3_L001_R1_001.fastq.gz" DC-IL-1.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-IL-2_S4_L001_R1_001.fastq.gz" DC-IL-2.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-WT-1_S5_L001_R1_001.fastq.gz" DC-WT-1.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-WT-2_S6_L001_R1_001.fastq.gz" DC-WT-2.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-OT-1_S7_L001_R1_001.fastq.gz" DC-OT-1.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-OT-2_S8_L001_R1_001.fastq.gz" DC-OT-2.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-IL-OT-CD69H-1_S9_L001_R1_001.fastq.gz" DC-IL-OT-CD69H-1.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-IL-OT-CD69H-2_S10_L001_R1_001.fastq.gz" DC-IL-OT-CD69H-2.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-IL-OT-CD69L-1_S11_L001_R1_001.fastq.gz" DC-IL-OT-CD69L-1.fastq.gz
ln -sf "$ROOT/data/raw/fastq/DC-IL-OT-CD69L-2_S12_L001_R1_001.fastq.gz" DC-IL-OT-CD69L-2.fastq.gz

# Recovered MAGeCK commands.
mageck count -l OTI.txt -n OTI --sample-label DC-1,DC-2,DC-IL-1,DC-IL-2,WT-1,WT-2,OT-1,OT-2,OT-CD69H-1,OT-CD69H-2,OT-CD69L-1,OT-CD69L-2 --fastq DC-1.fastq.gz DC-2.fastq.gz DC-IL-1.fastq.gz DC-IL-2.fastq.gz DC-WT-1.fastq.gz DC-WT-2.fastq.gz DC-OT-1.fastq.gz DC-OT-2.fastq.gz DC-IL-OT-CD69H-1.fastq.gz DC-IL-OT-CD69H-2.fastq.gz DC-IL-OT-CD69L-1.fastq.gz DC-IL-OT-CD69L-2.fastq.gz
mageck test -k OTI.count.txt -t OT-1 -c DC-1,DC-2 -n OT-1
mageck test -k OTI.count.txt -t OT-2 -c DC-1,DC-2 -n OT-2
mageck test -k OTI.count.txt -t OT-1,OT-2 -c DC-1,DC-2 -n OT
mageck test -k OTI.count.txt -t WT-1 -c DC-1,DC-2 -n WT-1
mageck test -k OTI.count.txt -t WT-2 -c DC-1,DC-2 -n WT-2
mageck test -k OTI.count.txt -t WT-1,WT-2 -c DC-1,DC-2 -n WT
