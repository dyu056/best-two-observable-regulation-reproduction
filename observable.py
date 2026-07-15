"""Stable observable collection helpers for autoresearch training.

Keep this file out of the model-edit loop: train.py should import and use the
register, while architecture experiments can add probe calls without changing
the serialization and logging protocol.
"""

from __future__ import annotations

import csv
import math
import re
import shutil
from pathlib import Path
from typing import Any


class ObservableRegister:
    """Collect scalar probes during one training step, then flush and clear."""

    def __init__(self) -> None:
        self.enabled = True
        self.context = ""
        self._values: dict[str, float] = {}
        self.records: list[dict[str, Any]] = []

    def clear(self) -> None:
        self._values.clear()

    def set_context(self, context: str) -> None:
        self.context = context.strip(".")

    def add(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        self._store_scalar(name, self._to_float(value))

    def add_l1_l2_abs_max(self, prefix: str, tensor: Any) -> None:
        if not self.enabled:
            return
        try:
            t = tensor.detach().float()
            self._store_scalar(f"{prefix}.l1", self._to_float(t.abs().sum()))
            self._store_scalar(f"{prefix}.l2", self._to_float(t.square().sum().sqrt()))
            self._store_scalar(f"{prefix}.abs_max", self._to_float(t.abs().max()))
        except Exception:
            return

    def add_norms(self, prefix: str, tensor: Any) -> None:
        self.add_l1_l2_abs_max(prefix, tensor)

    def flush(self) -> dict[str, float]:
        values = dict(self._values)
        self.clear()
        return values

    def record_step(self, step: int, epoch: int | None, values: dict[str, float]) -> None:
        clean_values = {
            key: float(value)
            for key, value in values.items()
            if isinstance(key, str) and isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        self.records.append({
            "step": int(step),
            "epoch": int(epoch) if epoch is not None else -1,
            "values": clean_values,
        })

    def curves(self) -> dict[str, list[dict[str, float]]]:
        series: dict[str, list[dict[str, float]]] = {}
        for record in self.records:
            step = record["step"]
            for key, value in record["values"].items():
                series.setdefault(key, []).append({"step": step, "value": value})
        return series

    def write_curves_csvs(self, directory: str | Path, metadata: dict[str, Any] | None = None) -> None:
        directory = Path(directory)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with (directory / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key, value in _flatten_metadata(metadata or {}):
                writer.writerow([key, value])

        series: dict[str, list[tuple[int, int, float]]] = {}
        for record in self.records:
            step = int(record["step"])
            epoch = int(record.get("epoch", -1))
            for key, value in record["values"].items():
                series.setdefault(key, []).append((step, epoch, float(value)))

        manifest_rows = []
        used_filenames: set[str] = set()
        for name, points in sorted(series.items()):
            filename = _unique_csv_name(_csv_safe_name(name), used_filenames)
            manifest_rows.append([name, filename, len(points)])
            with (directory / filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "epoch", "value"])
                writer.writerows(points)

        with (directory / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["observable", "filename", "num_points"])
            writer.writerows(manifest_rows)

    def _store_scalar(self, name: str, scalar: float | None) -> None:
        if scalar is None or not math.isfinite(scalar):
            return
        if self.context:
            name = f"{self.context}.{name}"
        self._values[name] = scalar

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "float"):
                value = value.float()
            if hasattr(value, "mean") and getattr(value, "ndim", 0) != 0:
                value = value.mean()
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
        except Exception:
            return None


OBS = ObservableRegister()


def _csv_safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "unnamed"


def _unique_csv_name(stem: str, used: set[str]) -> str:
    filename = f"{stem}.csv"
    if filename not in used:
        used.add(filename)
        return filename
    suffix = 2
    while True:
        filename = f"{stem}_{suffix}.csv"
        if filename not in used:
            used.add(filename)
            return filename
        suffix += 1


def _flatten_metadata(metadata: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in sorted(metadata.items()):
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_metadata(value, full_key))
        else:
            rows.append((full_key, str(value)))
    return rows


def build_step_observables(
    *,
    progress: float,
    train_loss: float,
    val_loss: float | None,
    smoothed_train_loss: float | None = None,
    train_accuracy: float | None = None,
    val_accuracy: float | None = None,
    gradient_norm: float | None = None,
    weight_observables: dict[str, float] | None = None,
    extra_observables: dict[str, float] | None = None,
    dt: float,
    tok_per_sec: int,
    lrm_muon: float,
    lrm_adam: float,
    muon_momentum: float,
    muon_weight_decay: float,
    mfu: float,
) -> dict[str, float]:
    # ===================== RSI OBSERVABILITY SYSTEM: METRICS BEGIN =====================
    # Strong marker for humans/AI:
    # These names are the stable dashboard/logging contract. Rename only if the
    # dashboard parser and any downstream analysis are updated at the same time.
    """Default step-level probes; add stable global variables here."""
    metrics = {
        "progress": progress,
        "raw_train_loss": train_loss,
        "train_val_gap": train_loss - val_loss if val_loss is not None else 0.0,
        "observable_train_test_gap": val_loss - train_loss if val_loss is not None else 0.0,
        "tok_per_sec_k": tok_per_sec / 1000.0,
        "step_time_ms": dt * 1000.0,
        "lrm_muon": lrm_muon,
        "lrm_adam": lrm_adam,
        "muon_momentum": muon_momentum,
        "muon_weight_decay": muon_weight_decay,
        "mfu_percent": mfu,
    }
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    if smoothed_train_loss is not None:
        metrics["observable_smoothed_train_loss"] = smoothed_train_loss
    if train_accuracy is not None:
        metrics["observable_accuracy.train"] = train_accuracy
    if val_accuracy is not None:
        metrics["observable_accuracy.test"] = val_accuracy
    if gradient_norm is not None:
        metrics["observable_gradient_norm"] = gradient_norm
    if weight_observables is not None:
        metrics.update(weight_observables)
    if extra_observables is not None:
        metrics.update(extra_observables)
    # ====================== RSI OBSERVABILITY SYSTEM: METRICS END ======================
    return metrics


def format_observable_line(step: int, epoch: int | None, values: dict[str, float]) -> str:
    # ===================== RSI OBSERVABILITY SYSTEM: LOG LINE BEGIN =====================
    # Legacy log compatibility only. train.py now records observables in memory and writes
    # per-observable CSV files after training instead of printing one line per step.
    payload = " ".join(f"{key}={value:.6g}" for key, value in sorted(values.items()))
    line = f"observable: step={step} epoch={epoch if epoch is not None else -1} {payload}"
    # ====================== RSI OBSERVABILITY SYSTEM: LOG LINE END ======================
    return line
