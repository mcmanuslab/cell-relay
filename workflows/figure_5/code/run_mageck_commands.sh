#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAGECK_DIR="$ROOT/data/upstream/mageck"
WORKDIR="$ROOT/data/upstream/mageck_rerun"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
cp "$MAGECK_DIR/NT466.count.txt" ./
cp "$MAGECK_DIR/design_matrix.txt" ./
cp "$MAGECK_DIR/design_matrix_split.txt" ./

# Recovered MAGeCK commands with available inputs.
mageck test -k NT466.count.txt -t B -c K -n B
# Skipped, missing recorded inputs: mageck mle -k NT466.count.txt -d design_matrix_b.txt -n B_mle
mageck test -k NT466.count.txt -t C -c K -n C
mageck mle -k NT466.count.txt -d design_matrix.txt -n NT466_mle
mageck mle -k NT466.count.txt -d design_matrix_split.txt -n NT466_mle_split
