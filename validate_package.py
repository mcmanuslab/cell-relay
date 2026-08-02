#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "figure_1f": 2,
    "figure_2i": 1,
    "figure_3b": 1,
    "figure_3d_f": 3,
    "figure_4": 28,
    "figure_5": 34,
    "extended_data_figure_4": 8,
    "extended_data_figure_8": 6,
}
PROHIBITED_SUFFIXES = (
    ".fastq",
    ".fastq.gz",
    ".fq",
    ".fq.gz",
    ".bam",
    ".bai",
    ".fcs",
    ".cloupe",
    ".h5",
    ".h5ad",
    ".sqlite",
    ".tar.gz",
    ".zip",
)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".r", ".sh", ".tsv", ".txt", ".yaml", ".yml"}
SCAN_EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "mplconfig", "outputs"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate included inputs and optional generated outputs.")
    parser.add_argument(
        "--require-outputs",
        action="store_true",
        help="Require all expected generated figures and passing quality summaries.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    scan_paths = [path for path in ROOT.rglob("*") if not SCAN_EXCLUDED_PARTS.intersection(path.parts)]
    all_files = [path for path in scan_paths if path.is_file()]
    large = [path for path in all_files if path.stat().st_size > 50 * 1024 * 1024]
    failures.extend(f"File exceeds 50 MiB: {path.relative_to(ROOT)}" for path in large)

    prohibited = [
        path for path in (ROOT / "workflows").rglob("*")
        if path.is_file() and path.name.lower().endswith(PROHIBITED_SUFFIXES)
    ]
    failures.extend(f"Prohibited large-data format: {path.relative_to(ROOT)}" for path in prohibited)

    outputs_present = any((ROOT / "workflows" / name / "outputs").exists() for name in EXPECTED)
    check_outputs = args.require_outputs or outputs_present

    for name, expected_count in EXPECTED.items():
        workflow = ROOT / "workflows" / name
        manifest_path = workflow / "data_manifest.tsv"
        if not manifest_path.exists():
            failures.append(f"Missing data manifest: {name}")
        else:
            with manifest_path.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    data_path = workflow / row["relative_path"]
                    if not data_path.exists():
                        failures.append(f"Missing manifested data: {name}/{row['relative_path']}")
                    elif sha256(data_path) != row["sha256"]:
                        failures.append(f"Checksum mismatch: {name}/{row['relative_path']}")

        if check_outputs:
            quality_path = workflow / "outputs" / "summaries" / "quality_summary.json"
            if not quality_path.exists():
                failures.append(f"Missing quality summary: {name}")
                continue
            quality = json.loads(quality_path.read_text())
            if quality.get("status") != "pass":
                failures.append(f"Quality summary did not pass: {name}")
            if quality.get("pdf_count") != expected_count or quality.get("png_count") != expected_count:
                failures.append(f"Unexpected figure count: {name}")

    private_tokens = ("/Users/", "/Volumes/Warp/", "Figure_data_audit")
    for path in all_files:
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Makefile":
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if any(token in text for token in private_tokens):
            failures.append(f"Machine-specific path in text file: {path.relative_to(ROOT)}")

    result = {
        "status": "pass" if not failures else "fail",
        "workflow_count": len(EXPECTED),
        "expected_pdf_png_pairs": sum(EXPECTED.values()),
        "outputs_checked": check_outputs,
        "largest_file_bytes": max((path.stat().st_size for path in all_files), default=0),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
