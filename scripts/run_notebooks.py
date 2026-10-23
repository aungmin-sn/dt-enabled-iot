"""Execute selected notebooks in fresh Python kernels; fail on any cell error."""
from pathlib import Path
import argparse
import asyncio
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="Notebook numeric prefixes, e.g. 01 02")
    parser.add_argument("--skip-simulation", action="store_true", help="Explicitly omit external solvers; reported as not run")
    args = parser.parse_args()
    validation = ROOT / ".validation"
    validation.mkdir(exist_ok=True)
    report_path = validation / ("execution_" + ("-".join(args.only) if args.only else "all") + ".json")
    kernels = validation / "kernels" / "repository-python"
    kernels.mkdir(parents=True, exist_ok=True)
    (kernels / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Repository validation Python", "language": "python"}), encoding="utf-8")
    manager = KernelSpecManager(kernel_dirs=[str(kernels.parent)])
    report = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if args.only and path.name[:2] not in args.only:
            continue
        if args.skip_simulation and path.name.startswith("05"):
            report.append({"notebook": path.name, "status": "NOT_RUN", "reason": "External solver execution explicitly omitted"})
            continue
        print("RUN", path.name, flush=True)
        started = time.monotonic()
        nb = nbformat.read(path, as_version=4)
        kernel = AsyncKernelManager(kernel_name="repository-python", kernel_spec_manager=manager)
        client = NotebookClient(nb, km=kernel, timeout=7200, kernel_name="repository-python",
                                resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        try:
            client.execute(cleanup_kc=True)
        except Exception as error:
            report.append({"notebook": path.name, "status": "FAILED", "error": str(error)[-2500:]})
            nbformat.write(nb, validation / path.name)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise
        nbformat.write(nb, validation / path.name)
        nbformat.write(nb, path)
        report.append({"notebook": path.name, "status": "PASSED", "code_cells": sum(c.cell_type == "code" for c in nb.cells),
                       "seconds": round(time.monotonic()-started, 2)})
        print("PASS", path.name, report[-1]["seconds"], "s", flush=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
