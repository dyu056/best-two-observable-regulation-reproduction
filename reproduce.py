#!/usr/bin/env python3
"""Reproduce the best completed two-observable trial (anchor-1 row 404).

The script launches the bundled training source with the exact seed, step count,
trajectory targets, coefficients, lead, and cutoff used by the winning sweep
trial.  It deliberately writes into a fresh work directory so generated files
cannot affect a later rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


SOURCE_FILES = ("train.py", "lib.py", "observable.py", "prepare.py", "data_split.json")


def main() -> None:
    bundle = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python3")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--output-dir", type=Path, default=bundle / "rerun")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(bundle / name, work / name)
    shutil.rmtree(work / "run_artifacts", ignore_errors=True)
    shutil.rmtree(work / "observable_csv", ignore_errors=True)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "MAX_TRAIN_STEPS": "2000",
            "SEED": "42",
            "LR_SCALE": "1.0",
            "WEIGHT_NORM_TARGET_REG": "0",
            "MLP_L2_GROWTH_REG": "0",
            "ATTN_V_HEAD_L2_GROWTH_REG": "0",
            "ATTN_QKV_HEAD_L2_GROWTH_REG": "0",
            "ATTN_ENTROPY_REG": "0",
            "OBSERVE_LAYER_PROBES": "0",
            "OBSERVE_REST100_ATTENTION_HEAD_L2": "0",
            "OBSERVE_HESSIAN_EIGENVALUES": "0",
            "WRITE_OBSERVABLE_ARTIFACTS": "0",
            "WRITE_OBSERVABLE_PLOTS": "0",
            "PRINT_OBSERVABLE_LINES": "0",
            "TRAJECTORY_REG_OBSERVABLE": "val.layer_2.attn_out.l1",
            "TRAJECTORY_REG_CURVE_FILE": str(bundle / "curves" / "val.layer_2.attn_out.l1.csv"),
            "TRAJECTORY_REG_MODE": "trajectory",
            "TRAJECTORY_REG_COEF": "0.01",
            "TRAJECTORY_REG_OBSERVABLE_2": "train.layer_0.k.l1",
            "TRAJECTORY_REG_CURVE_FILE_2": str(bundle / "curves" / "train.layer_0.k.l1.csv"),
            "TRAJECTORY_REG_MODE_2": "trajectory",
            "TRAJECTORY_REG_COEF_2": "0.01",
            "TRAJECTORY_REG_UNTIL_STEP": "750",
            "TRAJECTORY_REG_LEAD_STEPS": "100",
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
