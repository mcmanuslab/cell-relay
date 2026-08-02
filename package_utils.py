from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEPENDENCIES = ["adjustText", "matplotlib", "numpy", "pandas", "regex", "scipy"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def expand_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        else:
            raise FileNotFoundError(path)
    return sorted(set(files))


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def audit_pdf_fonts(path: Path, root: Path) -> dict[str, Any]:
    relative_path = path.resolve().relative_to(root.resolve()).as_posix()
    pdffonts = shutil.which("pdffonts")
    if pdffonts:
        result = subprocess.run([pdffonts, str(path)], capture_output=True, text=True)
        output = result.stdout + result.stderr
        lower = output.lower()
        type3 = "type 3" in lower
        truetype = "truetype" in lower
        return {
            "path": relative_path,
            "method": "pdffonts",
            "font_objects_found": bool(output.strip()) and result.returncode == 0,
            "truetype_font_objects_found": truetype,
            "truetype_compatible": not type3 and (truetype or result.returncode == 0),
            "type3_fonts_found": type3,
            "font_listing_available": result.returncode == 0,
        }
    payload = path.read_bytes()
    type3 = b"/Subtype /Type3" in payload
    font_objects = b"/Type /Font" in payload
    truetype = any(
        marker in payload
        for marker in [b"/Subtype /TrueType", b"/Subtype /CIDFontType2", b"/FontFile2"]
    )
    return {
        "path": relative_path,
        "method": "pdf_byte_scan",
        "font_objects_found": font_objects,
        "truetype_font_objects_found": truetype,
        "truetype_compatible": not type3 and (truetype or not font_objects),
        "type3_fonts_found": type3,
        "font_listing_available": False,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def portable_command(workflow_root: Path) -> list[str]:
    package_root = workflow_root.parents[1]
    args = []
    for value in sys.argv:
        path = Path(value)
        if path.is_absolute():
            try:
                value = path.relative_to(package_root).as_posix()
            except ValueError:
                value = path.name
        args.append(value)
    return ["python", *args]


def finish_run(
    workflow_root: Path,
    workflow_name: str,
    started_at: float,
    input_paths: Iterable[Path],
    *,
    details: dict[str, Any] | None = None,
    expected_files: Iterable[str] = (),
) -> dict[str, Any]:
    workflow_root = workflow_root.resolve()
    figure_dir = workflow_root / "outputs" / "figures"
    summary_dir = workflow_root / "outputs" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    inputs = [file_record(path, workflow_root) for path in expand_files(input_paths)]
    figure_files = sorted(path for path in figure_dir.rglob("*") if path.is_file())
    outputs = [file_record(path, workflow_root) for path in figure_files]
    pdfs = sorted(path for path in figure_files if path.suffix.lower() == ".pdf")
    pngs = sorted(path for path in figure_files if path.suffix.lower() == ".png")
    pdf_audit = [audit_pdf_fonts(path, workflow_root) for path in pdfs]

    pdf_stems = {path.relative_to(figure_dir).with_suffix("").as_posix() for path in pdfs}
    png_stems = {path.relative_to(figure_dir).with_suffix("").as_posix() for path in pngs}
    missing = [item for item in expected_files if not (workflow_root / item).exists()]
    quality = {
        "workflow": workflow_name,
        "status": "pass",
        "checks": {
            "figure_files_present": bool(figure_files),
            "pdf_png_pairs_match": pdf_stems == png_stems,
            "pdf_fonts_truetype_compatible": all(row["truetype_compatible"] for row in pdf_audit),
            "type3_fonts_absent": not any(row["type3_fonts_found"] for row in pdf_audit),
            "expected_files_present": not missing,
        },
        "missing_expected_files": missing,
        "pdf_font_audit": pdf_audit,
        "pdf_count": len(pdfs),
        "png_count": len(pngs),
    }
    if not all(quality["checks"].values()):
        quality["status"] = "fail"

    summary = {
        "workflow": workflow_name,
        "status": "completed" if quality["status"] == "pass" else "completed_with_failed_checks",
        "started_utc": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "completed_utc": utc_now(),
        "runtime_seconds": round(time.time() - started_at, 3),
        "command": portable_command(workflow_root),
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "inputs": inputs,
        "figure_outputs": outputs,
        "quality_summary": "outputs/summaries/quality_summary.json",
        "details": details or {},
    }
    write_json(summary_dir / "quality_summary.json", quality)
    write_json(summary_dir / "reproduction_summary.json", summary)
    if quality["status"] != "pass":
        raise RuntimeError(f"Quality checks failed for {workflow_name}; see {summary_dir / 'quality_summary.json'}")
    return summary
