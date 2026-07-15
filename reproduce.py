#!/usr/bin/env python3
"""Reproduce the unregularized baseline used for pair-sweep comparison."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

SOURCE_FILES = ("train.py", "lib.py", "observable.py", "prepare.py", "data_split.json")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python3")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, default=root / "rerun")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(root / name, work / name)
    shutil.rmtree(work / "run_artifacts", ignore_errors=True)
    shutil.rmtree(work / "observable_csv", ignore_errors=True)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "MAX_TRAIN_STEPS": "2000",
            "SEED": "42",
            "ATTN_BACKEND": "sdpa",
            "WEIGHT_NORM_TARGET_REG": "0",
            "OBSERVE_LAYER_PROBES": "1",
            "OBSERVE_REST100_ATTENTION_HEAD_L2": "1",
            "OBSERVE_HESSIAN_EIGENVALUES": "0",
            "PRINT_OBSERVABLE_LINES": "0",
        }
    )

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [args.python, "-u", "train.py"],
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise SystemExit(f"training exited {result.returncode}; inspect {log_path}")

    generated = work / "run_artifacts" / "latest_rsi_training_summary.json"
    if not generated.exists():
        raise SystemExit(f"training succeeded but summary is missing: {generated}")
    summary = json.loads(generated.read_text(encoding="utf-8"))
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"val_bpb": summary["val_bpb"], "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
