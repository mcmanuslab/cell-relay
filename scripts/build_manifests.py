#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PACKAGE_PARTS = {".git", ".venv", "__pycache__", "mplconfig", "outputs"}
EXCLUDED_PACKAGE_FILES = {".DS_Store", ".RData", ".Rhistory", "run_all_summary.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for workflow in sorted(path for path in (ROOT / "workflows").iterdir() if path.is_dir()):
        data_dir = workflow / "data" / "figure_tables"
        rows = []
        for path in sorted(item for item in data_dir.rglob("*") if item.is_file()):
            relative = path.relative_to(workflow).as_posix()
            role = "small_source_asset" if "flow_panel_sources" in path.parts else "processed_figure_input"
            rows.append(
                {
                    "relative_path": relative,
                    "role": role,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        write_tsv(workflow / "data_manifest.tsv", ["relative_path", "role", "size_bytes", "sha256"], rows)

    package_rows = []
    package_manifest = ROOT / "package_file_manifest.tsv"
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        if (
            path == package_manifest
            or path.name in EXCLUDED_PACKAGE_FILES
            or EXCLUDED_PACKAGE_PARTS.intersection(path.parts)
        ):
            continue
        package_rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_tsv(package_manifest, ["relative_path", "size_bytes", "sha256"], package_rows)


if __name__ == "__main__":
    main()
