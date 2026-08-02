#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKFLOWS = [
    "figure_1f",
    "figure_2i",
    "figure_3b",
    "figure_3d_f",
    "figure_4",
    "figure_5",
    "extended_data_figure_4",
    "extended_data_figure_8",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate publication figures from included tables.")
    parser.add_argument(
        "--workflow",
        action="append",
        choices=WORKFLOWS,
        help="Run one workflow; repeat the option to select several. Default: all.",
    )
    args = parser.parse_args()
    selected = args.workflow or WORKFLOWS
    started = time.time()
    records = []

    for name in selected:
        script = ROOT / "workflows" / name / "code" / "recreate_figures.py"
        print(f"\n[{name}] {script.relative_to(ROOT)}", flush=True)
        step_started = time.time()
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        summary_path = ROOT / "workflows" / name / "outputs" / "summaries" / "reproduction_summary.json"
        quality_path = ROOT / "workflows" / name / "outputs" / "summaries" / "quality_summary.json"
        records.append(
            {
                "workflow": name,
                "return_code": result.returncode,
                "runtime_seconds": round(time.time() - step_started, 3),
                "summary": str(summary_path.relative_to(ROOT)),
                "quality_summary": str(quality_path.relative_to(ROOT)),
            }
        )
        if result.returncode != 0:
            break

    payload = {
        "status": "pass" if len(records) == len(selected) and all(row["return_code"] == 0 for row in records) else "fail",
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "python": sys.version,
        "workflows": records,
    }
    (ROOT / "run_all_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nRun status: {payload['status']}")
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
