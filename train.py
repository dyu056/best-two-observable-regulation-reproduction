# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
"""
Nanochat pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Mean validation BPB: 0.9109 (10 seeds).
Usage: uv run train.py
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import csv
import json
import math
import re
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

# Keep default inductor settings for this compile-capture ablation.
import torch.nn as nn
import torch.nn.functional as F

ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "sdpa").lower()
cap = torch.cuda.get_device_capability()
_FORCE_MATH_SDPA = False

if ATTN_BACKEND == "sdpa":
    # H200 fast fix: avoid the flash-attention native extension path, which can
    # segfault before Python can report a traceback. This keeps the same data,
    # model scale, and causal attention shape while using PyTorch's CUDA SDPA.
    def attention_func(q, k, v, causal=True, window_size=(-1, -1)):
        q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if _FORCE_MATH_SDPA:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            ):
                return F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal).transpose(1, 2)
        return F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal).transpose(1, 2)

    print(f"Using PyTorch SDPA attention (GPU capability {cap})")
elif cap[0] >= 10:
    # Blackwell (B200, SM100): wrap flash-attn-4 as a custom op so torch.compile
    # treats it as opaque (no tracing into cutlass DSL, no recompile-cache thrash,
    # no per-call Python kernel build).
    from flash_attn.cute import flash_attn_func as _fa4_raw
    from flash_attn.cute.interface import _flash_attn_bwd as _fa4_bwd_raw

    @torch.library.custom_op("fa4::fa4_causal", mutates_args=())
    def _fa4_causal_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
        ws = (window_left, 0) if window_left > 0 else (None, None)
        out, lse = _fa4_raw(q, k, v, causal=True, window_size=ws, return_lse=True)
        return out, lse

    @_fa4_causal_op.register_fake
    def _fa4_causal_fake(q, k, v, window_left):
        B, T, H, D = q.shape
        return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)

    def _fa4_setup_context(ctx, inputs, output):
        q, k, v, window_left = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window_left = window_left

    @torch.library.custom_op("fa4::fa4_bwd", mutates_args=())
    def _fa4_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor, grad_output: torch.Tensor, lse: torch.Tensor,
                    window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wl = window_left if window_left > 0 else None
        dq, dk, dv = _fa4_bwd_raw(
            q, k, v, out, grad_output, lse,
            causal=True, window_size_left=wl, window_size_right=0,
        )
        return dq, dk, dv

    @_fa4_bwd_op.register_fake
    def _fa4_bwd_fake(q, k, v, out, grad_output, lse, window_left):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fa4_backward(ctx, grad_output, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.fa4.fa4_bwd(q, k, v, out, grad_output, lse, ctx.window_left)
        return dq, dk, dv, None

    _fa4_causal_op.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)

    def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
        wl = window_size[0] if isinstance(window_size, tuple) else window_size
        if wl is None or wl <= 0 or wl >= q.shape[1]:
            wl = -1
        out, _lse = torch.ops.fa4.fa4_causal(q, k, v, wl)
        return out

    attention_func = flash_attn_func
    print(f"Using flash-attn-4 as custom op (GPU capability {cap})")
else:
    # Previous default for Hopper/Ampere (H100, H200, A100): flash-attn-3 via
    # kernels package. Keep it available with ATTN_BACKEND=fa3 for later perf work.
    from kernels import get_kernel

    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    flash_attn_func = get_kernel(repo).flash_attn_interface.flash_attn_func
    attention_func = flash_attn_func
    print(f"Using flash-attn-3 from {repo} (GPU capability {cap})")

from lib import (  # noqa: E402
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    evaluate_bpb,
    load_data_split,
    make_dataloader,
    split_parquet_files,
)
from observable import OBS, build_step_observables, format_observable_line  # noqa: E402

RUN_ARTIFACT_DIR = Path("run_artifacts")
OBSERVABLE_CSV_DIR = Path("observable_csv")
WRITE_OBSERVABLE_ARTIFACTS = os.environ.get("WRITE_OBSERVABLE_ARTIFACTS", "1") == "1"
WRITE_OBSERVABLE_PLOTS = os.environ.get("WRITE_OBSERVABLE_PLOTS", "1") == "1"

TRAJECTORY_REG_OBSERVABLE = os.environ.get("TRAJECTORY_REG_OBSERVABLE", "").strip()
TRAJECTORY_REG_CURVE_FILE = os.environ.get("TRAJECTORY_REG_CURVE_FILE", "").strip()
TRAJECTORY_REG_MODE = os.environ.get("TRAJECTORY_REG_MODE", "trajectory").strip()
TRAJECTORY_REG_COEF = float(os.environ.get("TRAJECTORY_REG_COEF", "0.01"))
TRAJECTORY_REG_UNTIL_STEP = int(os.environ.get("TRAJECTORY_REG_UNTIL_STEP", "750"))
TRAJECTORY_REG_LEAD_STEPS = int(os.environ.get("TRAJECTORY_REG_LEAD_STEPS", "100"))
TRAJECTORY_REG_OBSERVABLE_2 = os.environ.get("TRAJECTORY_REG_OBSERVABLE_2", "").strip()
TRAJECTORY_REG_CURVE_FILE_2 = os.environ.get("TRAJECTORY_REG_CURVE_FILE_2", "").strip()
TRAJECTORY_REG_MODE_2 = os.environ.get("TRAJECTORY_REG_MODE_2", "trajectory").strip()
TRAJECTORY_REG_COEF_2 = float(os.environ.get("TRAJECTORY_REG_COEF_2", "0.01"))


def _canonical_reg_observable(name):
    for prefix in ("train.", "val."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


class _TrajectoryRegCapture:
    """Hold the one differentiable activation selected for this trial."""

    def __init__(self, observable):
        self.observable = _canonical_reg_observable(observable)
        self.value = None

    @property
    def enabled(self):
        return bool(self.observable) and self.observable.startswith("layer_")

    def clear(self):
        self.value = None

    def wants(self, name):
        return self.enabled and self.observable == _canonical_reg_observable(name)

    def wants_layer(self, layer_idx):
        return self.enabled and self.observable.startswith(f"layer_{layer_idx}.")

    def add(self, name, value):
        if self.wants(name):
            self.value = value.float().mean() if value.ndim else value.float()

    def add_norms(self, prefix, tensor):
        if self.wants(f"{prefix}.l1"):
            self.value = tensor.float().abs().sum()
        elif self.wants(f"{prefix}.l2"):
            self.value = tensor.float().square().sum().clamp_min(1e-12).sqrt()
        elif self.wants(f"{prefix}.abs_max"):
            self.value = tensor.float().abs().max()


REG_CAPTURE = _TrajectoryRegCapture(TRAJECTORY_REG_OBSERVABLE)
REG_CAPTURE_2 = _TrajectoryRegCapture(TRAJECTORY_REG_OBSERVABLE_2)
REG_CAPTURES = (REG_CAPTURE, REG_CAPTURE_2)
ATTENTION_ENTROPY_REG_VALUES = []
ATTENTION_ENTROPY_REG_CAPTURE_ACTIVE = False


def _load_trajectory_targets(path):
    if not path:
        return [], 1.0
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(float(row["value"]))
    if not rows:
        raise ValueError(f"Trajectory regularizer curve is empty: {path}")
    finite = torch.tensor(rows, dtype=torch.float64)
    scale = max(float(finite.std().item()), float(finite.abs().median().item()) * 0.05, 1e-6)
    return rows, scale


TRAJECTORY_REG_TARGETS, TRAJECTORY_REG_SCALE = _load_trajectory_targets(TRAJECTORY_REG_CURVE_FILE)
TRAJECTORY_REG_TARGETS_2, TRAJECTORY_REG_SCALE_2 = _load_trajectory_targets(TRAJECTORY_REG_CURVE_FILE_2)


def write_observable_artifacts(summary):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    RUN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"summary": summary}

    summary_path = RUN_ARTIFACT_DIR / f"rsi_training_summary_{stamp}.json"
    latest_summary_path = RUN_ARTIFACT_DIR / "latest_rsi_training_summary.json"
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    summary_path.write_text(summary_json + "\n", encoding="utf-8")
    latest_summary_path.write_text(summary_json + "\n", encoding="utf-8")

    if not WRITE_OBSERVABLE_ARTIFACTS:
        return Path("disabled"), Path("disabled"), Path("disabled")

    csv_dir = RUN_ARTIFACT_DIR / f"rsi_observable_csv_{stamp}"
    latest_csv_dir = RUN_ARTIFACT_DIR / "latest_rsi_observable_csv"
    for output in (csv_dir, latest_csv_dir, OBSERVABLE_CSV_DIR):
        OBS.write_curves_csvs(output, metadata=metadata)

    plot_path = RUN_ARTIFACT_DIR / f"rsi_observable_plot_{stamp}.png"
    latest_plot_path = RUN_ARTIFACT_DIR / "latest_rsi_observable_plot.png"
    if not WRITE_OBSERVABLE_PLOTS:
        return csv_dir, Path("disabled"), Path("disabled")
    _write_observable_plot(plot_path)
    shutil.copy2(plot_path, latest_plot_path)
    comparison_dir = RUN_ARTIFACT_DIR / f"rsi_observable_comparisons_{stamp}"
    latest_comparison_dir = RUN_ARTIFACT_DIR / "latest_rsi_observable_comparisons"
    _write_observable_comparison_plots(comparison_dir)
    _write_layer_weight_plots(comparison_dir)
    _write_hessian_eigenvalue_plot(comparison_dir)
    if latest_comparison_dir.exists():
        shutil.rmtree(latest_comparison_dir)
    shutil.copytree(comparison_dir, latest_comparison_dir)

    figure_index_path = RUN_ARTIFACT_DIR / f"rsi_observable_figure_index_{stamp}.csv"
    latest_figure_index_path = RUN_ARTIFACT_DIR / "latest_rsi_observable_figure_index.csv"
    _write_observable_figure_index(
        figure_index_path,
        csv_dir=csv_dir,
        latest_csv_dir=latest_csv_dir,
        comparison_dir=comparison_dir,
        latest_comparison_dir=latest_comparison_dir,
    )
    shutil.copy2(figure_index_path, latest_figure_index_path)
    return csv_dir, plot_path, figure_index_path


def _write_observable_plot(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = OBS.curves()
    preferred = [
        "raw_train_loss",
        "val_loss",
        "train_val_gap",
        "tok_per_sec_k",
        "mfu_percent",
        "lrm_muon",
        "lrm_adam",
        "muon_weight_decay",
    ]
    keys = [key for key in preferred if key in curves]
    keys.extend(key for key in sorted(curves) if key not in keys)
    keys = keys[:12]

    if not keys:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No observable records collected", ha="center", va="center")
        ax.axis("off")
    else:
        ncols = 2
        nrows = math.ceil(len(keys) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, 2.6 * nrows)), squeeze=False)
        for ax, key in zip(axes.ravel(), keys):
            points = curves[key]
            ax.plot([point["step"] for point in points], [point["value"] for point in points], linewidth=1.2)
            ax.set_title(key)
            ax.grid(True, alpha=0.25)
        for ax in axes.ravel()[len(keys):]:
            ax.axis("off")
        fig.suptitle("Training Observables", y=0.995)
        fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_observable_comparison_plots(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = OBS.curves()
    train_points = curves.get("raw_train_loss")
    val_points = curves.get("val_loss")
    if not train_points or not val_points:
        return

    directory.mkdir(parents=True, exist_ok=True)
    loss_keys = {"raw_train_loss", "val_loss"}
    for key in sorted(k for k in curves if k not in loss_keys):
        points = curves[key]
        if not points:
            continue
        safe_key = _observable_safe_name(key)
        fig, ax_obs = plt.subplots(figsize=(11, 5.5))
        ax_loss = ax_obs.twinx()
        ax_obs.plot(
            [point["step"] for point in points],
            [point["value"] for point in points],
            color="tab:green",
            linewidth=1.3,
            label=key,
        )
        ax_loss.plot(
            [point["step"] for point in train_points],
            [point["value"] for point in train_points],
            color="tab:blue",
            linewidth=1.0,
            alpha=0.75,
            label="raw_train_loss",
        )
        ax_loss.plot(
            [point["step"] for point in val_points],
            [point["value"] for point in val_points],
            color="tab:orange",
            linewidth=1.0,
            alpha=0.75,
            label="val_loss",
        )
        ax_obs.set_title(f"{key} vs train/test loss")
        ax_obs.set_xlabel("step")
        ax_obs.set_ylabel(key, color="tab:green")
        ax_loss.set_ylabel("loss")
        ax_obs.grid(True, alpha=0.25)
        lines, labels = ax_obs.get_legend_handles_labels()
        loss_lines, loss_labels = ax_loss.get_legend_handles_labels()
        ax_obs.legend(lines + loss_lines, labels + loss_labels, loc="best")
        fig.tight_layout()
        fig.savefig(directory / f"{safe_key}_vs_loss.png", dpi=160)
        plt.close(fig)


def _write_observable_figure_index(path, *, csv_dir, latest_csv_dir, comparison_dir, latest_comparison_dir):
    curves = OBS.curves()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "observable",
                "num_points",
                "csv_file",
                "latest_csv_file",
                "figure_file",
                "latest_figure_file",
                "figure_exists",
            ],
        )
        writer.writeheader()
        for key in sorted(curves):
            safe_key = _observable_safe_name(key)
            figure_file = comparison_dir / f"{safe_key}_vs_loss.png"
            latest_figure_file = latest_comparison_dir / f"{safe_key}_vs_loss.png"
            writer.writerow({
                "observable": key,
                "num_points": len(curves[key]),
                "csv_file": str(csv_dir / f"{safe_key}.csv"),
                "latest_csv_file": str(latest_csv_dir / f"{safe_key}.csv"),
                "figure_file": str(figure_file),
                "latest_figure_file": str(latest_figure_file),
                "figure_exists": figure_file.exists(),
            })


def _observable_safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._") or "observable"


def _write_layer_weight_plots(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = OBS.curves()
    train_points = curves.get("raw_train_loss")
    val_points = curves.get("val_loss")
    if not train_points or not val_points:
        return

    directory.mkdir(parents=True, exist_ok=True)
    for prefix, title, filename in (
        ("observable_weight_l1.layer_", "Layer Weight L1 vs train/test loss", "observable_weight_l1.layers_vs_loss.png"),
        ("observable_weight_l2.layer_", "Layer Weight L2 vs train/test loss", "observable_weight_l2.layers_vs_loss.png"),
    ):
        keys = sorted(k for k in curves if k.startswith(prefix))
        if not keys:
            continue
        fig, ax_weight = plt.subplots(figsize=(12, 6))
        ax_loss = ax_weight.twinx()
        cmap = plt.get_cmap("tab10")
        for index, key in enumerate(keys):
            points = curves[key]
            label = key.removeprefix(prefix)
            ax_weight.plot(
                [point["step"] for point in points],
                [point["value"] for point in points],
                linewidth=1.2,
                color=cmap(index % 10),
                label=label,
            )
        ax_loss.plot(
            [point["step"] for point in train_points],
            [point["value"] for point in train_points],
            color="black",
            linewidth=1.0,
            alpha=0.65,
            label="raw_train_loss",
        )
        ax_loss.plot(
            [point["step"] for point in val_points],
            [point["value"] for point in val_points],
            color="tab:orange",
            linewidth=1.0,
            alpha=0.75,
            label="val_loss",
        )
        ax_weight.set_title(title)
        ax_weight.set_xlabel("step")
        ax_weight.set_ylabel("weight norm")
        ax_loss.set_ylabel("loss")
        ax_weight.grid(True, alpha=0.25)
        weight_lines, weight_labels = ax_weight.get_legend_handles_labels()
        loss_lines, loss_labels = ax_loss.get_legend_handles_labels()
        ax_weight.legend(
            weight_lines + loss_lines,
            weight_labels + loss_labels,
            loc="center left",
            bbox_to_anchor=(1.08, 0.5),
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)


def _write_hessian_eigenvalue_plot(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = OBS.curves()
    keys = [
        "observable_hessian_eigenvalues.scalar_min",
        "observable_hessian_eigenvalues.scalar_max",
        "observable_hessian_eigenvalues.scalar_abs_max",
        "observable_hessian_eigenvalues.scalar_trace",
    ]
    keys = [key for key in keys if key in curves]
    train_points = curves.get("raw_train_loss")
    val_points = curves.get("val_loss")
    if not keys or not train_points or not val_points:
        return

    directory.mkdir(parents=True, exist_ok=True)
    fig, ax_hessian = plt.subplots(figsize=(11, 5.5))
    ax_loss = ax_hessian.twinx()
    colors = {
        "observable_hessian_eigenvalues.scalar_min": "tab:red",
        "observable_hessian_eigenvalues.scalar_max": "tab:purple",
        "observable_hessian_eigenvalues.scalar_abs_max": "tab:green",
        "observable_hessian_eigenvalues.scalar_trace": "tab:brown",
    }
    for key in keys:
        points = curves[key]
        ax_hessian.plot(
            [point["step"] for point in points],
            [point["value"] for point in points],
            linewidth=1.4,
            color=colors.get(key),
            label=key,
        )
    ax_loss.plot(
        [point["step"] for point in train_points],
        [point["value"] for point in train_points],
        color="tab:blue",
        linewidth=1.0,
        alpha=0.7,
        label="raw_train_loss",
    )
    ax_loss.plot(
        [point["step"] for point in val_points],
        [point["value"] for point in val_points],
        color="tab:orange",
        linewidth=1.0,
        alpha=0.7,
        label="val_loss",
    )
    ax_hessian.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax_hessian.set_title("Scalar-control Hessian eigenvalues vs train/test loss")
    ax_hessian.set_xlabel("step")
    ax_hessian.set_ylabel("Hessian eigenvalue")
    ax_loss.set_ylabel("loss")
    ax_hessian.grid(True, alpha=0.25)
    hessian_lines, hessian_labels = ax_hessian.get_legend_handles_labels()
    loss_lines, loss_labels = ax_loss.get_legend_handles_labels()
    ax_hessian.legend(hessian_lines + loss_lines, hessian_labels + loss_labels, loc="best")
    fig.tight_layout()
    fig.savefig(directory / "observable_hessian_eigenvalues.scalar_vs_loss.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def compute_layer_weight_observables(module):
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    grouped: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, param in module.named_parameters():
        group = _weight_observable_group(name)
        if group is None:
            continue
        p = param.detach().float()
        if group not in grouped:
            grouped[group] = (
                torch.zeros((), device=p.device, dtype=torch.float32),
                torch.zeros((), device=p.device, dtype=torch.float32),
            )
        weight_l1, weight_l2_sq = grouped[group]
        grouped[group] = (weight_l1 + p.abs().sum(), weight_l2_sq + p.square().sum())

    values = {}
    for group, (weight_l1, weight_l2_sq) in grouped.items():
        values[f"observable_weight_l1.{group}"] = weight_l1.item()
        values[f"observable_weight_l2.{group}"] = weight_l2_sq.sqrt().item()
    return values


@torch.no_grad()
def compute_attention_parameter_weight_observables(module):
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    values = {}
    for name, param in module.named_parameters():
        group = _attention_qkv_parameter_group(name)
        if group is None:
            continue
        p = param.detach().float()
        finite = torch.isfinite(p)
        safe = torch.where(finite, p, torch.zeros_like(p))
        values[f"observable_param_abs_max.{group}"] = safe.abs().max().item()
        values[f"observable_param_l1.{group}"] = safe.abs().sum().item()
        values[f"observable_param_l2.{group}"] = safe.square().sum().sqrt().item()
        values[f"observable_param_mean.{group}"] = safe.mean().item()
        values[f"observable_param_rms.{group}"] = safe.square().mean().sqrt().item()
        values[f"observable_param_std.{group}"] = safe.std(unbiased=False).item()
        values[f"observable_param_nan_fraction.{group}"] = torch.isnan(p).float().mean().item()
        values[f"observable_param_zero_fraction.{group}"] = (p == 0).float().mean().item()
    return values


@torch.no_grad()
def compute_mlp_weight_observables(module):
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    values = {}
    for layer_idx, block in enumerate(module.transformer.h):
        l1_total = torch.zeros((), device=next(block.mlp.parameters()).device, dtype=torch.float32)
        l2_sq_total = torch.zeros((), device=l1_total.device, dtype=torch.float32)
        for name, param in (
            ("c_fc", block.mlp.c_fc.weight),
            ("c_proj", block.mlp.c_proj.weight),
        ):
            p = param.detach().float()
            tensor_l1 = p.abs().sum()
            tensor_l2 = p.square().sum().sqrt()
            l1_total = l1_total + tensor_l1
            l2_sq_total = l2_sq_total + p.square().sum()
            values[f"observable_param_l1.layer_{layer_idx:02d}.mlp.{name}.weight"] = tensor_l1.item()
            values[f"observable_param_l2.layer_{layer_idx:02d}.mlp.{name}.weight"] = tensor_l2.item()
        values[f"observable_mlp_weight_l1.layer_{layer_idx:02d}"] = l1_total.item()
        values[f"observable_mlp_weight_l2.layer_{layer_idx:02d}"] = l2_sq_total.sqrt().item()
    return values


REST100_ATTENTION_HEAD_L2_SELECTION = tuple(
    (layer_idx, proj, head_idx)
    for layer_idx in range(8)
    for head_idx in range(4)
    for proj in ("c_q", "c_k", "c_v")
) + (
    (5, "c_v", 4),
    (5, "c_v", 5),
    (7, "c_v", 4),
    (7, "c_v", 5),
)


@torch.no_grad()
def compute_rest100_attention_head_l2_observables(module):
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    selected = set(REST100_ATTENTION_HEAD_L2_SELECTION)
    values = {}
    for layer_idx, block in enumerate(module.transformer.h):
        attn = block.attn
        for proj in ("c_q", "c_k", "c_v"):
            param = getattr(attn, proj).weight.detach().float()
            head_count = attn.n_head if proj == "c_q" else attn.n_kv_head
            head_dim = attn.head_dim
            head_weights = param.view(head_count, head_dim, -1)
            for head_idx in range(head_count):
                if (layer_idx, proj, head_idx) not in selected:
                    continue
                safe = torch.nan_to_num(head_weights[head_idx])
                values[
                    f"observable_param_head_l2.layer_{layer_idx:02d}.attn.{proj}.head_{head_idx:02d}.weight"
                ] = safe.square().sum().sqrt().item()
    return values


def _weight_observable_group(name):
    parts = name.split(".")
    for idx in range(len(parts) - 2):
        if parts[idx] == "transformer" and parts[idx + 1] == "h" and parts[idx + 2].isdigit():
            return f"layer_{int(parts[idx + 2]):02d}"
    for idx in range(len(parts) - 1):
        if parts[idx] in {"value_embeds", "bigram_ves", "trigram_ves"} and parts[idx + 1].isdigit():
            return f"layer_{int(parts[idx + 1]):02d}.{parts[idx]}"
    return None


def _attention_qkv_parameter_group(name):
    parts = name.split(".")
    for idx in range(len(parts) - 5):
        if (
            parts[idx] == "transformer"
            and parts[idx + 1] == "h"
            and parts[idx + 2].isdigit()
            and parts[idx + 3] == "attn"
            and parts[idx + 4] in {"c_q", "c_k", "c_v"}
            and parts[idx + 5] == "weight"
        ):
            return f"layer_{int(parts[idx + 2]):02d}.attn.{parts[idx + 4]}.weight"
    return None


def compute_weight_norm_target_regularizer(module):
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    grouped: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        group = _weight_observable_group(name)
        if group is None:
            continue
        p = param.float()
        group_l2_sq = p.square().sum()
        grouped[group] = grouped.get(group, torch.zeros((), device=p.device, dtype=torch.float32)) + group_l2_sq

    if not grouped:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {}

    target = torch.tensor(WEIGHT_NORM_TARGET, device=device, dtype=torch.float32)
    norms = torch.stack([value.clamp_min(1e-12).sqrt() for value in grouped.values()])
    relative_errors = (norms - target) / target.clamp_min(1.0)
    deficits = (target - norms).clamp_min(0.0) / target.clamp_min(1.0)
    penalty = WEIGHT_NORM_TARGET_REG_COEF * relative_errors.square().mean()
    metrics = {
        "observable_weight_norm_target_reg.loss": penalty.detach().item(),
        "observable_weight_norm_target_reg.target": WEIGHT_NORM_TARGET,
        "observable_weight_norm_target_reg.mean_l2": norms.detach().mean().item(),
        "observable_weight_norm_target_reg.min_l2": norms.detach().min().item(),
        "observable_weight_norm_target_reg.max_l2": norms.detach().max().item(),
        "observable_weight_norm_target_reg.mean_abs_relative_error": relative_errors.detach().abs().mean().item(),
        "observable_weight_norm_target_reg.mean_deficit": deficits.detach().mean().item(),
        "observable_weight_norm_target_reg.num_under_target": (norms.detach() < target).float().sum().item(),
        "observable_weight_norm_target_reg.num_over_target": (norms.detach() > target).float().sum().item(),
    }
    return penalty, metrics


def _linear_ramp_target(step, until_step, start, end):
    if until_step <= 0:
        return end
    alpha = min(max(step / until_step, 0.0), 1.0)
    return start + alpha * (end - start)


def compute_mlp_l2_growth_regularizer(module, step):
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    target = _linear_ramp_target(
        step,
        MLP_L2_GROWTH_REG_UNTIL_STEP,
        MLP_L2_GROWTH_REG_START_TARGET,
        MLP_L2_GROWTH_REG_END_TARGET,
    )
    target_t = torch.tensor(target, device=device, dtype=torch.float32)
    penalties = []
    norms = []
    selected_layers = set(MLP_L2_GROWTH_REG_LAYERS)
    for layer_idx, block in enumerate(module.transformer.h):
        if layer_idx not in selected_layers:
            continue
        l2_sq = torch.zeros((), device=target_t.device, dtype=torch.float32)
        for param in (block.mlp.c_fc.weight, block.mlp.c_proj.weight):
            l2_sq = l2_sq + param.float().square().sum()
        norm_value = l2_sq.clamp_min(1e-12).sqrt()
        norms.append(norm_value)
        penalties.append(((target_t - norm_value).clamp_min(0.0) / target_t.clamp_min(1.0)).square())

    if not penalties:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {}

    norms_t = torch.stack(norms)
    penalty = MLP_L2_GROWTH_REG_COEF * torch.stack(penalties).mean()
    metrics = {
        "observable_reg_mlp_l2_growth.loss": penalty.detach().item(),
        "observable_reg_mlp_l2_growth.target": float(target),
        "observable_reg_mlp_l2_growth.mean_l2": norms_t.detach().mean().item(),
        "observable_reg_mlp_l2_growth.min_l2": norms_t.detach().min().item(),
        "observable_reg_mlp_l2_growth.max_l2": norms_t.detach().max().item(),
        "observable_reg_mlp_l2_growth.num_under_target": (norms_t.detach() < target_t).float().sum().item(),
    }
    return penalty, metrics


def compute_attention_head_l2_growth_regularizer(module, step, projections, prefix):
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    target = _linear_ramp_target(
        step,
        ATTN_HEAD_L2_GROWTH_REG_UNTIL_STEP,
        ATTN_HEAD_L2_GROWTH_REG_START_TARGET,
        ATTN_HEAD_L2_GROWTH_REG_END_TARGET,
    )
    target_t = torch.tensor(target, device=device, dtype=torch.float32)
    selected_layers = set(ATTN_HEAD_L2_GROWTH_REG_LAYERS)
    penalties = []
    norms = []
    for layer_idx, block in enumerate(module.transformer.h):
        if layer_idx not in selected_layers:
            continue
        attn = block.attn
        for proj in projections:
            param = getattr(attn, proj).weight.float()
            head_count = attn.n_head if proj == "c_q" else attn.n_kv_head
            head_dim = attn.head_dim
            head_weights = param.view(head_count, head_dim, -1)
            head_norms = head_weights.square().sum(dim=(1, 2)).clamp_min(1e-12).sqrt()
            norms.append(head_norms)
            penalties.append(((target_t - head_norms).clamp_min(0.0) / target_t.clamp_min(1.0)).square())

    if not penalties:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {}

    norms_t = torch.cat(norms)
    penalty = ATTN_HEAD_L2_GROWTH_REG_COEF * torch.cat(penalties).mean()
    metrics = {
        f"observable_reg_{prefix}.loss": penalty.detach().item(),
        f"observable_reg_{prefix}.target": float(target),
        f"observable_reg_{prefix}.mean_l2": norms_t.detach().mean().item(),
        f"observable_reg_{prefix}.min_l2": norms_t.detach().min().item(),
        f"observable_reg_{prefix}.max_l2": norms_t.detach().max().item(),
        f"observable_reg_{prefix}.num_under_target": (norms_t.detach() < target_t).float().sum().item(),
    }
    return penalty, metrics


def compute_attention_entropy_regularizer():
    if not ATTENTION_ENTROPY_REG_VALUES:
        raise ValueError("Attention entropy regularizer captured no layer values")
    if len(ATTENTION_ENTROPY_REG_VALUES) != DEPTH:
        raise ValueError(
            f"Attention entropy regularizer expected {DEPTH} layers, "
            f"captured {len(ATTENTION_ENTROPY_REG_VALUES)}"
        )
    entropies = torch.stack(ATTENTION_ENTROPY_REG_VALUES)
    penalty = ATTN_ENTROPY_REG_COEF * entropies.mean()
    return penalty, {
        "observable_reg_attention_entropy.loss": penalty.detach().item(),
        "observable_reg_attention_entropy.mean": entropies.detach().mean().item(),
        "observable_reg_attention_entropy.min": entropies.detach().min().item(),
        "observable_reg_attention_entropy.max": entropies.detach().max().item(),
        "observable_reg_attention_entropy.layer_count": float(entropies.numel()),
    }


def _tensor_observable_stat(tensor, statistic):
    value = tensor.float()
    if statistic == "l1":
        return value.abs().sum()
    if statistic == "l2":
        return value.square().sum().clamp_min(1e-12).sqrt()
    if statistic == "abs_max":
        return value.abs().max()
    if statistic == "mean":
        return value.mean()
    if statistic == "rms":
        return value.square().mean().clamp_min(1e-12).sqrt()
    if statistic == "std":
        return value.std()
    if statistic == "zero_fraction":
        scale = value.detach().square().mean().sqrt().clamp_min(1e-6) * 0.01
        return torch.exp(-value.abs() / scale).mean()
    if statistic == "nan_fraction":
        # A finite run has no differentiable NaN count; this zero-valued barrier
        # keeps the trial explicit without introducing NaNs into the objective.
        return torch.nan_to_num(value).sum() * 0.0
    raise ValueError(f"Unsupported trajectory statistic: {statistic}")


def compute_parameter_trajectory_value(module, observable):
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    observable = _canonical_reg_observable(observable)

    match = re.fullmatch(
        r"observable_param_(l1|l2|abs_max|mean|rms|std|zero_fraction|nan_fraction)\."
        r"layer_(\d+)\.attn\.(c_q|c_k|c_v)\.weight",
        observable,
    )
    if match:
        statistic, layer, projection = match.groups()
        param = getattr(module.transformer.h[int(layer)].attn, projection).weight
        return _tensor_observable_stat(param, statistic)

    match = re.fullmatch(
        r"observable_param_(l1|l2)\.layer_(\d+)\.mlp\.(c_fc|c_proj)\.weight",
        observable,
    )
    if match:
        statistic, layer, projection = match.groups()
        param = getattr(module.transformer.h[int(layer)].mlp, projection).weight
        return _tensor_observable_stat(param, statistic)

    match = re.fullmatch(
        r"observable_param_head_l2\.layer_(\d+)\.attn\.(c_q|c_k|c_v)\.head_(\d+)\.weight",
        observable,
    )
    if match:
        layer, projection, head = match.groups()
        attn = module.transformer.h[int(layer)].attn
        param = getattr(attn, projection).weight.float()
        head_count = attn.n_head if projection == "c_q" else attn.n_kv_head
        head_weights = param.view(head_count, attn.head_dim, -1)
        return _tensor_observable_stat(head_weights[int(head)], "l2")

    match = re.fullmatch(r"observable_mlp_weight_(l1|l2)\.layer_(\d+)", observable)
    if match:
        statistic, layer = match.groups()
        mlp = module.transformer.h[int(layer)].mlp
        flat = torch.cat((mlp.c_fc.weight.float().flatten(), mlp.c_proj.weight.float().flatten()))
        return _tensor_observable_stat(flat, statistic)

    match = re.fullmatch(r"observable_weight_(l1|l2)\.layer_(\d+)(?:\.(\w+))?", observable)
    if match:
        statistic, layer, component = match.groups()
        target_group = f"layer_{int(layer):02d}" + (f".{component}" if component else "")
        tensors = [
            param.float()
            for name, param in module.named_parameters()
            if _weight_observable_group(name) == target_group
        ]
        if tensors:
            if statistic == "l1":
                return torch.stack([tensor.abs().sum() for tensor in tensors]).sum()
            return torch.stack([tensor.square().sum() for tensor in tensors]).sum().clamp_min(1e-12).sqrt()
    return None


def compute_trajectory_regularizer(module, step, task_loss):
    if not TRAJECTORY_REG_OBSERVABLE or step > TRAJECTORY_REG_UNTIL_STEP:
        return task_loss.new_zeros(()), {}
    if TRAJECTORY_REG_MODE == "loss_acceleration":
        penalty = TRAJECTORY_REG_COEF * task_loss
        return penalty, {
            "observable_trajectory_reg.loss": penalty.detach().item(),
            "observable_trajectory_reg.proxy": 1.0,
        }

    current = REG_CAPTURE.value
    if current is None:
        current = compute_parameter_trajectory_value(module, TRAJECTORY_REG_OBSERVABLE)
    if current is None:
        raise ValueError(f"No differentiable trajectory value for {TRAJECTORY_REG_OBSERVABLE}")
    if not TRAJECTORY_REG_TARGETS:
        raise ValueError("TRAJECTORY_REG_CURVE_FILE is required for trajectory mode")
    target_idx = min(step + TRAJECTORY_REG_LEAD_STEPS, len(TRAJECTORY_REG_TARGETS) - 1)
    target = current.new_tensor(TRAJECTORY_REG_TARGETS[target_idx])
    scale = current.new_tensor(TRAJECTORY_REG_SCALE).clamp_min(1e-6)
    penalty = TRAJECTORY_REG_COEF * ((current - target) / scale).square()
    observables = {
        "observable_trajectory_reg.loss": penalty.detach().item(),
        "observable_trajectory_reg.current": current.detach().item(),
        "observable_trajectory_reg.target": target.detach().item(),
        "observable_trajectory_reg.scale": float(TRAJECTORY_REG_SCALE),
    }
    if TRAJECTORY_REG_OBSERVABLE_2 and step <= TRAJECTORY_REG_UNTIL_STEP:
        if TRAJECTORY_REG_MODE_2 == "loss_acceleration":
            penalty_2 = TRAJECTORY_REG_COEF_2 * task_loss
            observables["observable_trajectory_reg_2.proxy"] = 1.0
        else:
            current_2 = REG_CAPTURE_2.value
            if current_2 is None:
                current_2 = compute_parameter_trajectory_value(module, TRAJECTORY_REG_OBSERVABLE_2)
            if current_2 is None:
                raise ValueError(f"No differentiable trajectory value for {TRAJECTORY_REG_OBSERVABLE_2}")
            if not TRAJECTORY_REG_TARGETS_2:
                raise ValueError("TRAJECTORY_REG_CURVE_FILE_2 is required for second trajectory mode")
            target_idx_2 = min(step + TRAJECTORY_REG_LEAD_STEPS, len(TRAJECTORY_REG_TARGETS_2) - 1)
            target_2 = current_2.new_tensor(TRAJECTORY_REG_TARGETS_2[target_idx_2])
            scale_2 = current_2.new_tensor(TRAJECTORY_REG_SCALE_2).clamp_min(1e-6)
            penalty_2 = TRAJECTORY_REG_COEF_2 * ((current_2 - target_2) / scale_2).square()
            observables.update({
                "observable_trajectory_reg_2.current": current_2.detach().item(),
                "observable_trajectory_reg_2.target": target_2.detach().item(),
                "observable_trajectory_reg_2.scale": float(TRAJECTORY_REG_SCALE_2),
            })
        observables["observable_trajectory_reg_2.loss"] = penalty_2.detach().item()
        penalty = penalty + penalty_2
        observables["observable_trajectory_reg.loss"] = penalty.detach().item()
    return penalty, observables


@torch.no_grad()
def compute_gradient_norm(module):
    grad_l2_sq = torch.zeros((), device=device, dtype=torch.float32)
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        grad_l2_sq += grad.square().sum()
    return grad_l2_sq.sqrt().item()


def compute_scalar_hessian_observables(module, x_hessian, y_hessian):
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    x_hessian = x_hessian[:HESSIAN_BATCH_SIZE, :HESSIAN_SEQ_LEN]
    y_hessian = y_hessian[:HESSIAN_BATCH_SIZE, :HESSIAN_SEQ_LEN]
    params = [
        module.resid_lambdas,
        module.x0_lambdas,
        module.x0_gate_scales,
        module.layer_pool_weights,
    ]
    global _FORCE_MATH_SDPA
    _FORCE_MATH_SDPA = True
    try:
        loss = module(x_hessian, y_hessian)
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            allow_unused=True,
        )
    finally:
        _FORCE_MATH_SDPA = False
    flat_grads = []
    flat_params = []
    for param, grad in zip(params, grads):
        if grad is None:
            continue
        flat_params.append(param.reshape(-1))
        flat_grads.append(grad.reshape(-1))
    if not flat_grads:
        return {}

    flat_grad = torch.cat(flat_grads)
    flat_param = torch.cat(flat_params)
    rows = []
    for grad_i in flat_grad:
        row_grads = torch.autograd.grad(
            grad_i,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        row_parts = []
        for param, row_grad in zip(params, row_grads):
            if row_grad is None:
                row_parts.append(torch.zeros_like(param).reshape(-1))
            else:
                row_parts.append(row_grad.reshape(-1))
        rows.append(torch.cat(row_parts))

    hessian = torch.stack(rows).float()
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    return {
        "observable_hessian_eigenvalues.scalar_min": eigenvalues.min().item(),
        "observable_hessian_eigenvalues.scalar_max": eigenvalues.max().item(),
        "observable_hessian_eigenvalues.scalar_abs_max": eigenvalues.abs().max().item(),
        "observable_hessian_eigenvalues.scalar_trace": hessian.diagonal().sum().item(),
        "observable_hessian_eigenvalues.scalar_neg_count": (eigenvalues < 0).sum().item(),
        "observable_hessian_eigenvalues.scalar_dim": flat_param.numel(),
        "observable_hessian_eigenvalues.batch_size": HESSIAN_BATCH_SIZE,
        "observable_hessian_eigenvalues.sequence_len": HESSIAN_SEQ_LEN,
    }

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def observe_attention_distribution(prefix, q, k, window_size, sample_tokens=64):
    """Sampled causal attention entropy probe for layer diagnostics."""
    try:
        track_grad = ATTENTION_ENTROPY_REG_CAPTURE_ACTIVE or any(
            capture.wants(f"{prefix}.{suffix}")
            for capture in REG_CAPTURES
            for suffix in (
                "attn_entropy",
                "attn_entropy_norm",
                "attn_max_prob",
                "sink_attention_mass",
                "attn_head_sink_max",
                "attn_norm",
                "attn_outlier_ratio",
            )
        )
        with nullcontext() if track_grad else torch.no_grad():
            q0 = q[0].float().transpose(0, 1) if track_grad else q[0].detach().float().transpose(0, 1)
            k0 = k[0].float().transpose(0, 1) if track_grad else k[0].detach().float().transpose(0, 1)
            if k0.size(0) != q0.size(0):
                repeat = q0.size(0) // k0.size(0)
                k0 = k0.repeat_interleave(repeat, dim=0)
            total_tokens = q0.size(1)
            q_count = min(sample_tokens, total_tokens)
            q_start = total_tokens - q_count
            q_sample = q0[:, q_start:, :]
            scores = torch.matmul(q_sample, k0.transpose(-2, -1)) / (q_sample.size(-1) ** 0.5)
            query_positions = torch.arange(q_start, total_tokens, device=scores.device)[:, None]
            key_positions = torch.arange(total_tokens, device=scores.device)[None, :]
            visible = key_positions <= query_positions
            left_window = window_size[0] if isinstance(window_size, tuple) else window_size
            if left_window is not None and left_window > 0 and left_window < total_tokens:
                visible = visible & (key_positions >= (query_positions - left_window + 1))
            scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            if ATTENTION_ENTROPY_REG_CAPTURE_ACTIVE:
                ATTENTION_ENTROPY_REG_VALUES.append(entropy.mean())
            support = visible.sum(dim=-1).float().clamp_min(1)
            sink_mass_by_head = probs[..., 0].mean(dim=-1)
            values = {
                "attn_entropy": entropy.mean(),
                "attn_entropy_norm": (entropy / support.log().clamp_min(1e-6)).mean(),
                "attn_max_prob": probs.max(dim=-1).values.mean(),
                "sink_attention_mass": sink_mass_by_head.mean(),
                "attn_head_sink_max": sink_mass_by_head.max(),
                "attn_norm": probs.square().sum(dim=-1).sqrt().mean(),
                "attn_outlier_ratio": (probs.max(dim=-1).values * support.unsqueeze(0)).mean(),
            }
            for suffix, value in values.items():
                if OBSERVE_LAYER_PROBES:
                    OBS.add(f"{prefix}.{suffix}", value)
                for capture in REG_CAPTURES:
                    capture.add(f"{prefix}.{suffix}", value)
    except Exception:
        return


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Separate gate for bigram VE on ALL VE layers reading decorrelated channels (32:64)
        self.bigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Trigram gate on layers 1, 5, and 7 for late/full-context coverage, reads channels 64:96
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        trigram_layers = (
            {ve_layers[0], ve_layers[-2], ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        self.trigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if layer_idx in trigram_layers
            else None
        )
        # Head-level MoE gate on ALL layers for attention output routing
        self.head_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size, bigram_ve=None, trigram_ve=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Bigram VE with its own independent gate reading from decorrelated channels (32:64)
        if bigram_ve is not None:
            bigram_ve = bigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            bg_gate = 2 * torch.sigmoid(self.bigram_gate(x[..., self.ve_gate_channels:2*self.ve_gate_channels]))
            v = v + bg_gate.unsqueeze(-1) * bigram_ve

        # Trigram VE with its own gate reading from channels 64:96
        if trigram_ve is not None:
            trigram_ve = trigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            tg_gate = 2 * torch.sigmoid(self.trigram_gate(x[..., 2*self.ve_gate_channels:3*self.ve_gate_channels]))
            v = v + tg_gate.unsqueeze(-1) * trigram_ve

        cos, sin = cos_sin
        # QK-norm refinement: normalize BEFORE rotary instead of after
        q, k = norm(q), norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if OBSERVE_LAYER_PROBES or any(c.wants_layer(self.layer_idx) for c in REG_CAPTURES) or ATTENTION_ENTROPY_REG_CAPTURE_ACTIVE:
            prefix = f"layer_{self.layer_idx}"
            if OBSERVE_LAYER_PROBES:
                OBS.add_norms(f"{prefix}.q", q)
                OBS.add_norms(f"{prefix}.k", k)
                OBS.add_norms(f"{prefix}.v", v)
            for capture in REG_CAPTURES:
                capture.add_norms(f"{prefix}.q", q)
                capture.add_norms(f"{prefix}.k", k)
                capture.add_norms(f"{prefix}.v", v)
            attention_probe_prefixes = (
                f"{prefix}.attn_",
                f"{prefix}.sink_attention_mass",
            )
            if (
                OBSERVE_LAYER_PROBES
                or any(c.observable.startswith(attention_probe_prefixes) for c in REG_CAPTURES)
                or ATTENTION_ENTROPY_REG_CAPTURE_ACTIVE
            ):
                observe_attention_distribution(prefix, q, k, window_size)

        # Replaces the previous direct flash_attn_func call; default SDPA is less
        # optimized but avoids the H200 native-extension segfault path.
        y = attention_func(q, k, v, causal=True, window_size=window_size)
        # Per-head RMSNorm on attention output (DiffTransformer-inspired sub-layer normalization)
        y = norm(y)

        # Head-level MoE: per-head routing gate on all layers
        head_gates = 2.0 * torch.sigmoid(self.head_gate(x[..., :self.ve_gate_channels]))
        if OBSERVE_LAYER_PROBES or any(c.wants(f"layer_{self.layer_idx}.head_gate_mean") for c in REG_CAPTURES):
            for capture in REG_CAPTURES:
                capture.add(f"layer_{self.layer_idx}.head_gate_mean", head_gates)
        if OBSERVE_LAYER_PROBES:
            OBS.add(f"layer_{self.layer_idx}.head_gate_mean", head_gates)
        y = y * head_gates.unsqueeze(-1)

        y = y.contiguous().view(B, T, -1)
        if OBSERVE_LAYER_PROBES:
            OBS.add_norms(f"layer_{self.layer_idx}.attn_out", y)
        for capture in REG_CAPTURES:
            capture.add_norms(f"layer_{self.layer_idx}.attn_out", y)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        # Uniform tau=0.5: confirmed optimal threshold
        self.tau = 0.5

    def forward(self, x):
        h_pre = self.c_fc(x)
        h_act = F.relu(h_pre - self.tau).square()
        h = self.c_proj(h_act)
        if OBSERVE_LAYER_PROBES:
            OBS.add_norms(f"layer_{self.layer_idx}.mlp_pre", h_pre)
            OBS.add_norms(f"layer_{self.layer_idx}.mlp_act", h_act)
            OBS.add_norms(f"layer_{self.layer_idx}.mlp_out", h)
        for capture in REG_CAPTURES:
            capture.add_norms(f"layer_{self.layer_idx}.mlp_pre", h_pre)
            capture.add_norms(f"layer_{self.layer_idx}.mlp_act", h_act)
            capture.add_norms(f"layer_{self.layer_idx}.mlp_out", h)
        return h


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

    def forward(self, x, ve, cos_sin, window_size, bigram_ve=None, trigram_ve=None):
        # Simplified attention residual: per-head norm + head gate inside CSA already sufficient
        x = x + self.attn(norm(x), ve, cos_sin, window_size, bigram_ve=bigram_ve, trigram_ve=trigram_ve)
        x = x + norm(self.mlp(norm(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # JEPA MTP removed: multi-token prediction hurts step count in 5-min budget
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Input-dependent x0 gating: per-layer scale for sigmoid gate on x0 skip (layers 4+)
        # gate = 2*sigmoid(scale * x.mean(-1)) modulates x0_lambdas contribution
        # Zero-init so gate starts at 1.0 (neutral = same as current scalar behavior)
        self.x0_gate_scales = nn.Parameter(torch.zeros(config.n_layer))
        # Multi-layer output pooling: aggregate last-K intermediate layers as additive correction
        self.n_pool_layers = min(4, config.n_layer)  # layers [n-4, n-3, n-2] contribute (3 weights)
        self.layer_pool_weights = nn.Parameter(torch.zeros(self.n_pool_layers - 1))
        # Value embeddings (unigram)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        # Factored multi-hash bigram VE: K=2 half-dim tables concatenated per layer
        # Crossover: K=2 simplification recovers throughput
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        self.bigram_ve_layers = set(ve_layers)
        self.bigram_table_size = config.vocab_size * 64  # CROSSOVER B: 64x bigram tables
        self.bigram_K = 2
        half_kv_dim = kv_dim // 2
        # PER-LAYER DECORRELATED: completely disjoint hash prime pairs per bigram VE layer
        # Each layer uses entirely distinct multipliers -- zero prime reuse within bigram type
        # Constants from Murmur/FNV/golden-ratio family for good avalanche behavior
        _decorr_bigram_primes = [
            [(2654435761, 2246822519), (1013904223, 6291469)],   # layer 1: golden-ratio family
            [(374761393, 668265263), (3266489917, 104729)],      # layer 3: prime family
            [(1640531527, 97531), (48271, 40503)],               # layer 5: LCG/Knuth family
            [(16777619, 2166136261), (3432918353, 461845907)],   # layer 7: MurmurHash3 family
        ]
        self.bigram_hash_primes_per_layer = {}
        self.bigram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(ve_layers):
            self.bigram_ves[str(layer_i)] = nn.ModuleList([
                nn.Embedding(self.bigram_table_size, half_kv_dim),
                nn.Embedding(self.bigram_table_size, half_kv_dim),
            ])
            self.bigram_hash_primes_per_layer[layer_i] = _decorr_bigram_primes[j]
        # Multi-layer factored trigram VE: K=2 half-dim tables at layers 1+5 plus layer 7.
        self.trigram_ve_layers = (
            {ve_layers[0], ve_layers[-2], ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        self.trigram_table_size = config.vocab_size * 64  # CROSSOVER B: 64x trigram tables
        # PER-LAYER DECORRELATED: completely disjoint 6-prime tuples per trigram VE layer
        # Using disjoint constant families: each layer uses different multiplier sources
        _decorr_trigram_primes = [
            (16777619, 2166136261, 3432918353, 461845907, 2654435769, 1540483477),  # layer 1: FNV+Murmur family
            (3405403843, 2654435761, 2246822519, 1013904223, 6291469, 374761393),   # layer 5: golden-ratio family
            (668265263, 3266489917, 104729, 1640531527, 97531, 48271),              # layer 7: prime family
        ]
        self.trigram_hash_primes_per_layer = {}
        self.trigram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(sorted(self.trigram_ve_layers)):
            self.trigram_ves[str(layer_i)] = nn.ModuleList([
                nn.Embedding(self.trigram_table_size, half_kv_dim),
                nn.Embedding(self.trigram_table_size, half_kv_dim),
            ])
            self.trigram_hash_primes_per_layer[layer_i] = _decorr_trigram_primes[j]
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        self.x0_gate_scales.fill_(0.0)  # Zero-init: sigmoid(0)=0.5, 2*0.5=1.0 = neutral gate
        self.layer_pool_weights.fill_(0.0)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.bigram_gate is not None:
                torch.nn.init.zeros_(block.attn.bigram_gate.weight)
            if block.attn.trigram_gate is not None:
                torch.nn.init.zeros_(block.attn.trigram_gate.weight)
            torch.nn.init.zeros_(block.attn.head_gate.weight)
        # Bigram VE: same init as regular VE (factored: two half-dim tables per layer)
        for layer_ves in self.bigram_ves.values():
            for bve in layer_ves:
                torch.nn.init.uniform_(bve.weight, -s, s)
                bve.to(dtype=torch.bfloat16)
        # Trigram VE init (factored: two half-dim tables per layer)
        for layer_tves in self.trigram_ves.values():
            for tve in layer_tves:
                torch.nn.init.uniform_(tve.weight, -s, s)
                tve.to(dtype=torch.bfloat16)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=1000000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SLT" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        tiny_window = long_window // 4
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0), "T": (tiny_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.layer_pool_weights.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        ngram_ve_betas=None,  # if None, uses adam_betas
        ngram_ve_lr_scale=1.0,  # discriminative LR scale for n-gram VE (ULMFiT-inspired)
    ):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas, self.x0_gate_scales]  # gate scales grouped with x0 lambdas
        bigram_ve_params = list(self.bigram_ves.parameters())
        trigram_ve_params = list(self.trigram_ves.parameters())
        pool_params = [self.layer_pool_weights]
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
            + len(bigram_ve_params)
            + len(trigram_ve_params)
            + len(pool_params)
        )
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if ngram_ve_betas is None:
            ngram_ve_betas = adam_betas
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            {
                "kind": "adamw",
                "params": lm_head_params,
                "lr": unembedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                "demon_beta1": True,  # Apply Demon beta1 scheduling
            },
            {
                "kind": "adamw",
                "params": embedding_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": value_embeds_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": resid_params,
                "lr": scalar_lr * 0.01,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                # No demon_beta1: scalar params keep fixed beta1
            },
            {
                "kind": "adamw",
                "params": x0_params,
                "lr": scalar_lr,
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.002,  # x0WD=0.002 (proven optimal)
                "is_x0_muon_warmdown": True,  # x0 Muon warmdown
            },
            {
                "kind": "rmsprop",
                "params": bigram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": 1e-10,
                "weight_decay": 0.0,
                "is_ngram_ve": True,
            },
            {
                "kind": "rmsprop",
                "params": trigram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": 1e-10,
                "weight_decay": 0.0,
                "is_ngram_ve": True,
            },
            {
                "kind": "adamw",
                "params": pool_params,
                "lr": scalar_lr * 0.15,  # revert to formula (0.75*0.15=0.1125)
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                {
                    "kind": "muon",
                    "params": group_params,
                    "lr": matrix_lr,
                    "momentum": 0.95,
                    "ns_steps": 5,
                    "beta2": 0.95,
                    "weight_decay": weight_decay,
                }
            )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean", return_accuracy=False):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        # PER-LAYER DECORRELATED: precompute shifted indices (shared), compute per-layer hash indices inside loop
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        # Precompute per-layer bigram hash indices (different primes per layer for collision decorrelation)
        bigram_indices_per_layer = {}
        for layer_i in self.bigram_ve_layers:
            layer_bg_primes = self.bigram_hash_primes_per_layer[layer_i]
            bigram_indices_per_layer[layer_i] = [
                ((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size
                for p1, p2 in layer_bg_primes
            ]
        # Precompute per-layer trigram hash indices (different primes per layer for collision decorrelation)
        trigram_indices_per_layer = {}
        for layer_i in self.trigram_ve_layers:
            lp = self.trigram_hash_primes_per_layer[layer_i]
            trigram_indices_per_layer[layer_i] = (
                ((prev2_idx * lp[0]) ^ (prev_idx * lp[1]) ^ (idx * lp[2])) % self.trigram_table_size,
                ((prev2_idx * lp[3]) ^ (prev_idx * lp[4]) ^ (idx * lp[5])) % self.trigram_table_size,
            )
        n_layer = len(self.transformer.h)
        pool_start = n_layer - self.n_pool_layers
        pool_residual = None
        for i, block in enumerate(self.transformer.h):
            # Input-dependent x0 gate on ALL 8 layers: 2*sigmoid(scale*mean(x)) modulates x0 contribution
            # Starts at 1.0 (gate_scales=0 → sigmoid(0)=0.5 → 2*0.5=1.0)
            x0_gate = 2.0 * torch.sigmoid(self.x0_gate_scales[i] * x.float().mean(-1, keepdim=True)).to(x.dtype)
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0_gate * x0
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            else:
                ve = None
            # Factored multi-hash bigram VE: concat K=2 half-dim lookups from independent hashes (per-layer primes)
            if i in self.bigram_ve_layers:
                layer_ves = self.bigram_ves[str(i)]
                layer_indices = bigram_indices_per_layer[i]
                bgve = torch.cat([layer_ves[k](layer_indices[k]) for k in range(self.bigram_K)], dim=-1)
            else:
                bgve = None
            # Multi-layer factored trigram VE: concat two half-dim lookups per layer (per-layer primes)
            if i in self.trigram_ve_layers:
                tg_idx = trigram_indices_per_layer[i]
                layer_tves = self.trigram_ves[str(i)]
                tgve = torch.cat([layer_tves[0](tg_idx[0]), layer_tves[1](tg_idx[1])], dim=-1)
            else:
                tgve = None
            x = block(x, ve, cos_sin, self.window_sizes[i], bigram_ve=bgve, trigram_ve=tgve)
            if i == pool_start:
                pool_residual = self.layer_pool_weights[0] * x
            elif i == pool_start + 1:
                pool_residual = pool_residual + self.layer_pool_weights[1] * x
            elif i == pool_start + 2:
                pool_residual = pool_residual + self.layer_pool_weights[2] * x
        if pool_residual is not None:
            x = x + pool_residual
        x = norm(x)

        # Decoupled softcap in BF16: skip float() cast, halve logit tensor memory
        # Since model is natively BF16, softcap in BF16 should be numerically adequate
        logits = self.lm_head(x)
        logits = 16.5 * torch.tanh(logits / 15.0)

        if targets is not None:
            # Cast to float32 only for the CE loss computation (numerically sensitive)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            if return_accuracy:
                with torch.no_grad():
                    valid = targets != -1
                    pred = logits.argmax(dim=-1)
                    correct = ((pred == targets) & valid).float().sum()
                    denom = valid.float().sum().clamp_min(1.0)
                    accuracy = correct / denom
                return loss, accuracy
            return loss
        # Eval path: need float32 logits
        return logits.float()


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


@torch.compile(dynamic=False, fullgraph=True)
def rmsprop_step_fused(p, grad, exp_avg_sq, step_t, lr_t, beta2_t, eps_t, wd_t):
    """RMSProp with bias correction -- no first moment, saves 50% optimizer VRAM."""
    p.mul_(1 - lr_t * wd_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(grad / denom, alpha=-lr_t)


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    ns_steps,
    red_dim,
):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # RMSProp CPU tensors (no beta1 -- saves first moment VRAM)
        self._rmsprop_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        step_fns = {
            "adamw": self._step_adamw,
            "rmsprop": self._step_rmsprop,
            "muon": self._step_muon,
        }
        self._step_dispatch = tuple((step_fns[group["kind"]], group) for group in self.param_groups)

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_rmsprop(self, group):
        """RMSProp: only second moment, no first moment -- 50% less optimizer VRAM for sparse tables."""
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg_sq"] = torch.zeros_like(p)
                # Note: NO exp_avg allocated -- this is the VRAM saving
            state["step"] += 1
            self._rmsprop_step_t.fill_(state["step"])
            self._rmsprop_lr_t.fill_(group["lr"])
            self._rmsprop_beta2_t.fill_(group["beta2"])
            self._rmsprop_eps_t.fill_(group["eps"])
            self._rmsprop_wd_t.fill_(group["weight_decay"])
            rmsprop_step_fused(
                p,
                grad,
                state["exp_avg_sq"],
                self._rmsprop_step_t,
                self._rmsprop_lr_t,
                self._rmsprop_beta2_t,
                self._rmsprop_eps_t,
                self._rmsprop_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for step_fn, group in self._step_dispatch:
            step_fn(group)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 96  # model_dim = depth * ASPECT_RATIO (d8*96=768 -> dim=768, 6 heads)
HEAD_DIM = 128  # target head dimension for attention
WINDOW_PATTERN = "TTTL"  # 3 tiny + 1 long -- sandwich norm + warmdown=0.8 variant

# Optimization
TOTAL_BATCH_SIZE = 72 * 2048  # 147456 tokens per step (grad_accum=1 with devbatch=72 on B200)
EMBEDDING_LR = 0.6  # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04  # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.8  # x0 Muon warmdown SCALAR_LR=0.8
WEIGHT_DECAY = 0.1  # baseline WD
ADAM_BETAS = (0.8, 0.95)  # Adam beta1, beta2
DEMON_FINAL_BETA1 = 0.55  # baseline Demon
NGRAM_VE_BETAS = (0.5, 0.999)  # RMSProp only uses beta2=0.999; higher beta2 preserves gradient history for sparse tables
NGRAM_VE_LR_SCALE = 1.0  # RMSProp with full LR (no reduction)
WARMUP_RATIO = 0.0  # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.95  # extend warmdown trend (0.80->0.85->0.90->0.95), warmdown starts at 5%
ADAM_WARMDOWN_RATIO = 0.65  # slightly longer Adam warmdown to match extended Muon warmdown
NGRAM_WARMDOWN_RATIO = 0.0  # no warmdown for bigram/trigram VE (sparse tables benefit from full-rate training)
FINAL_LR_FRAC = 0.05  # restored FLR=0.05

# Model size
DEPTH = 8  # number of transformer layers
DEVICE_BATCH_SIZE = 72  # per-device batch size -- B200
OBSERVE_LAYER_PROBES = os.environ.get("OBSERVE_LAYER_PROBES", "0") == "1"
OBSERVE_HESSIAN_EIGENVALUES = os.environ.get("OBSERVE_HESSIAN_EIGENVALUES", "0") == "1"
OBSERVE_REST100_ATTENTION_HEAD_L2 = os.environ.get("OBSERVE_REST100_ATTENTION_HEAD_L2", "0") == "1"
PRINT_OBSERVABLE_LINES = os.environ.get("PRINT_OBSERVABLE_LINES", "0") == "1"


def _parse_int_list_env(name, default):
    value = os.environ.get(name, default)
    if not value.strip():
        return tuple()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


HESSIAN_EVERY = int(os.environ.get("HESSIAN_EVERY", "200"))
if HESSIAN_EVERY <= 0:
    raise ValueError(f"HESSIAN_EVERY must be positive, got {HESSIAN_EVERY}")
HESSIAN_BATCH_SIZE = int(os.environ.get("HESSIAN_BATCH_SIZE", "1"))
HESSIAN_SEQ_LEN = int(os.environ.get("HESSIAN_SEQ_LEN", "128"))
if HESSIAN_BATCH_SIZE <= 0:
    raise ValueError(f"HESSIAN_BATCH_SIZE must be positive, got {HESSIAN_BATCH_SIZE}")
if HESSIAN_SEQ_LEN <= 0:
    raise ValueError(f"HESSIAN_SEQ_LEN must be positive, got {HESSIAN_SEQ_LEN}")
WEIGHT_NORM_TARGET_REG = os.environ.get("WEIGHT_NORM_TARGET_REG", "0") == "1"
WEIGHT_NORM_TARGET = float(os.environ.get("WEIGHT_NORM_TARGET", "350000"))
WEIGHT_NORM_TARGET_REG_UNTIL_STEP = int(os.environ.get("WEIGHT_NORM_TARGET_REG_UNTIL_STEP", "750"))
WEIGHT_NORM_TARGET_REG_COEF = float(os.environ.get("WEIGHT_NORM_TARGET_REG_COEF", "100"))
MLP_L2_GROWTH_REG = os.environ.get("MLP_L2_GROWTH_REG", "0") == "1"
MLP_L2_GROWTH_REG_LAYERS = _parse_int_list_env("MLP_L2_GROWTH_REG_LAYERS", "0,1,2")
MLP_L2_GROWTH_REG_UNTIL_STEP = int(os.environ.get("MLP_L2_GROWTH_REG_UNTIL_STEP", "750"))
MLP_L2_GROWTH_REG_COEF = float(os.environ.get("MLP_L2_GROWTH_REG_COEF", "1.0"))
MLP_L2_GROWTH_REG_START_TARGET = float(os.environ.get("MLP_L2_GROWTH_REG_START_TARGET", "55"))
MLP_L2_GROWTH_REG_END_TARGET = float(os.environ.get("MLP_L2_GROWTH_REG_END_TARGET", "165"))
ATTN_V_HEAD_L2_GROWTH_REG = os.environ.get("ATTN_V_HEAD_L2_GROWTH_REG", "0") == "1"
ATTN_QKV_HEAD_L2_GROWTH_REG = os.environ.get("ATTN_QKV_HEAD_L2_GROWTH_REG", "0") == "1"
ATTN_HEAD_L2_GROWTH_REG_LAYERS = _parse_int_list_env("ATTN_HEAD_L2_GROWTH_REG_LAYERS", "1,2,3,4,6")
ATTN_HEAD_L2_GROWTH_REG_UNTIL_STEP = int(os.environ.get("ATTN_HEAD_L2_GROWTH_REG_UNTIL_STEP", "750"))
ATTN_HEAD_L2_GROWTH_REG_COEF = float(os.environ.get("ATTN_HEAD_L2_GROWTH_REG_COEF", "1.0"))
ATTN_HEAD_L2_GROWTH_REG_START_TARGET = float(os.environ.get("ATTN_HEAD_L2_GROWTH_REG_START_TARGET", "11"))
ATTN_HEAD_L2_GROWTH_REG_END_TARGET = float(os.environ.get("ATTN_HEAD_L2_GROWTH_REG_END_TARGET", "28"))
ATTN_ENTROPY_REG = os.environ.get("ATTN_ENTROPY_REG", "0") == "1"
ATTN_ENTROPY_REG_UNTIL_STEP = int(os.environ.get("ATTN_ENTROPY_REG_UNTIL_STEP", "750"))
ATTN_ENTROPY_REG_COEF = float(os.environ.get("ATTN_ENTROPY_REG_COEF", "0.01"))
if WEIGHT_NORM_TARGET <= 0:
    raise ValueError(f"WEIGHT_NORM_TARGET must be positive, got {WEIGHT_NORM_TARGET}")
if WEIGHT_NORM_TARGET_REG_UNTIL_STEP < 0:
    raise ValueError(
        f"WEIGHT_NORM_TARGET_REG_UNTIL_STEP must be non-negative, got {WEIGHT_NORM_TARGET_REG_UNTIL_STEP}"
    )
for name, value in (
    ("MLP_L2_GROWTH_REG_UNTIL_STEP", MLP_L2_GROWTH_REG_UNTIL_STEP),
    ("ATTN_HEAD_L2_GROWTH_REG_UNTIL_STEP", ATTN_HEAD_L2_GROWTH_REG_UNTIL_STEP),
    ("ATTN_ENTROPY_REG_UNTIL_STEP", ATTN_ENTROPY_REG_UNTIL_STEP),
):
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
_SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
# No autocast: model is natively BF16 -- eliminates FP32->BF16 cast overhead in compile graph
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
B200_BF16_PEAK_FLOPS = 2.25e15

# Data check log: print the exact split and parquet files used by the
# train/val dataloaders before training starts.
data_split = load_data_split()
print(f"Data split train ids: {data_split['train']}", flush=True)
print(f"Data split test ids: {data_split['test']}", flush=True)
print(f"Train parquet files: {split_parquet_files('train')}", flush=True)
print(f"Test/val parquet files: {split_parquet_files('test')}", flush=True)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
# Cast entire model to BF16: enables removing autocast, simplifies compile graph
model.to(dtype=torch.bfloat16)

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    ngram_ve_betas=NGRAM_VE_BETAS,
    ngram_ve_lr_scale=NGRAM_VE_LR_SCALE,
)

muon_groups = []
ngram_groups = []
x0_warmdown_groups = []
adam_groups = []
adam_demon_groups = []
muon_group_lrs = []
x0_group_lrs = []
adam_group_lrs = []
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        muon_groups.append(group)
        muon_group_lrs.append((group, group["initial_lr"]))
    elif group.get("is_ngram_ve", False):
        ngram_groups.append(group)
    elif group.get("is_x0_muon_warmdown", False):
        x0_warmdown_groups.append(group)
        x0_group_lrs.append((group, group["initial_lr"]))
    else:
        adam_groups.append(group)
        adam_group_lrs.append((group, group["initial_lr"]))
        if group.get("demon_beta1", False):
            adam_demon_groups.append((group, group["betas"][1]))

if OBSERVE_LAYER_PROBES or OBSERVE_HESSIAN_EIGENVALUES or any(c.enabled for c in REG_CAPTURES) or ATTN_ENTROPY_REG:
    print("Observable Python/autograd probes enabled; torch.compile disabled for probe collection.")
else:
    model = torch.compile(model, dynamic=False, mode="max-autotune", fullgraph=True)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
val_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")
x, y, epoch = next(train_loader)  # prefetch first batch

MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "2000"))
if MAX_TRAIN_STEPS <= 0:
    raise ValueError(f"MAX_TRAIN_STEPS must be positive, got {MAX_TRAIN_STEPS}")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Max training steps: {MAX_TRAIN_STEPS}")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = step / MAX_TRAIN_STEPS)


def get_lr_multiplier(progress, warmdown_ratio=WARMDOWN_RATIO):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


MUON_PEAK_MOMENTUM = 0.95  # standard peak
MUON_WARMDOWN_MOMENTUM = 0.79  # testing VE beta2 ramp alone
# Reverse Demon for NorMuon beta2: INCREASE beta2 during warmdown for more stable variance normalization
MUON_BETA2_PEAK = 0.95  # standard beta2 during full-LR phase
MUON_BETA2_WARMDOWN = 0.97  # target beta2 at end of warmdown
MUON_LR_BOOST = 1.0  # no LR boost
# VE RMSProp reverse-Demon: increase VE beta2 during last 30% of Muon warmdown
# Analogous to Muon's 0.95->0.97, but for ngram VE tables (0.999->0.9995)
NGRAM_VE_BETA2_WARMDOWN = 0.9999  # STRONGER delayed VE beta2 ramp (0.999->0.9999 last 30% warmdown)
def get_muon_momentum(step, progress=None):
    # Warmup: 0.85 -> 0.95 over 300 steps
    frac = min(step / 300, 1)
    base = (1 - frac) * 0.85 + frac * MUON_PEAK_MOMENTUM
    # Quadratic Demon: back-loaded shape keeps peak momentum longer
    if progress is not None:
        warmdown_start = 1.0 - WARMDOWN_RATIO
        if progress > warmdown_start:
            wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
            base = MUON_PEAK_MOMENTUM + (wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    return base


def get_muon_beta2(progress):
    """Reverse beta2: increase beta2 during warmdown for more stable variance norm."""
    warmdown_start = 1.0 - WARMDOWN_RATIO
    if progress < warmdown_start:
        return MUON_BETA2_PEAK
    else:
        wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
        return MUON_BETA2_PEAK + wd_frac * (MUON_BETA2_WARMDOWN - MUON_BETA2_PEAK)


def get_muon_lr_boost(progress):
    """Boost Muon LR during warmdown to compensate for higher beta2 reducing step size."""
    warmdown_start = 1.0 - WARMDOWN_RATIO
    if progress < warmdown_start:
        return 1.0
    else:
        wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
        return 1.0 + wd_frac * (MUON_LR_BOOST - 1.0)


def get_adam_beta1(progress, warmdown_ratio=ADAM_WARMDOWN_RATIO):
    """Forward Demon: decrease beta1 during warmdown for more responsive gradient following."""
    initial_beta1 = ADAM_BETAS[0]
    final_beta1 = DEMON_FINAL_BETA1
    warmdown_start = 1.0 - warmdown_ratio
    if progress < warmdown_start:
        return initial_beta1
    else:
        warmdown_progress = (progress - warmdown_start) / warmdown_ratio
        return initial_beta1 + (final_beta1 - initial_beta1) * warmdown_progress


# WD pulse: RECTANGULAR shape -- with 95% warmdown (starts at 5%), pulses shifted earlier
# Main pulse at 3% center, 2% total duration (1% half-width): fires at 2-4%, before warmdown onset at 5%
# Early pulse at 1.5% center, 1% total duration: fires at 1-2% progress
# Both pulses fire in the full-LR phase (0-5%), maintaining the pre-warmdown regularization timing
WD_PULSE_CENTER = 0.03   # shift main pulse to 3% (fires before warmdown at 5%)
WD_PULSE_HALF_WIDTH = 0.01  # 1% half-width: 2% total duration (tighter for earlier firing)
WD_PULSE_MAGNITUDE = 5.0  # try 5x main pulse (vs 8x) -- 5x optimal WITH Muon Demon, 8x WITHOUT; current setup HAS Demon
WD_EARLY_PULSE_CENTER = 0.015  # shift early pulse to 1.5%
WD_EARLY_PULSE_HALF_WIDTH = 0.005  # 0.5% half-width: 1% total duration
WD_EARLY_PULSE_MAGNITUDE = 3.0  # 3x early pulse (gentler, to initialize regularization)
# Mid-warmdown triangular pulse: fires at 80% total progress (= ~79% through warmdown)
# This is WITHIN the VE beta2 ramp zone (which starts at 71.5% total = 70% through warmdown)
# Hypothesis: VE beta2 stabilization provides a safety net for a mid-warmdown WD perturbation
WD_MID_PULSE_CENTER = 0.80   # 80% total progress = ~79% through warmdown
WD_MID_PULSE_HALF_WIDTH = 0.025  # 2.5% half-width: 5% total triangular duration
WD_MID_PULSE_MAGNITUDE = 4.0  # 4x magnitude (triangular shape -- less harsh than rectangular)

def get_weight_decay(progress):
    base_wd = WEIGHT_DECAY * (1 - progress)
    # Early small pulse: 3x spike at 2% progress (step ~65), 2% total duration
    early_dist = abs(progress - WD_EARLY_PULSE_CENTER)
    if early_dist < WD_EARLY_PULSE_HALF_WIDTH:
        return base_wd * WD_EARLY_PULSE_MAGNITUDE  # RECTANGULAR early pulse
    # Main pulse: 8x rectangular spike at 5% progress (step ~163), 3% total duration
    dist = abs(progress - WD_PULSE_CENTER)
    if dist < WD_PULSE_HALF_WIDTH:
        return base_wd * WD_PULSE_MAGNITUDE  # RECTANGULAR main pulse
    # Mid-warmdown triangular pulse: fires within VE beta2 stabilization zone
    mid_dist = abs(progress - WD_MID_PULSE_CENTER)
    if mid_dist < WD_MID_PULSE_HALF_WIDTH:
        # Triangular: linear ramp up then down (proven optimal shape)
        local = (progress - (WD_MID_PULSE_CENTER - WD_MID_PULSE_HALF_WIDTH)) / (2 * WD_MID_PULSE_HALF_WIDTH)
        bump = 2 * local if local < 0.5 else 2 * (1 - local)
        return base_wd * (1.0 + bump * (WD_MID_PULSE_MAGNITUDE - 1.0))
    return base_wd


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
VAL_LOSS_EVERY = 1
inv_max_train_steps = 1.0 / MAX_TRAIN_STEPS
inv_muon_warmdown = 1.0 / WARMDOWN_RATIO
inv_adam_warmdown = 1.0 / ADAM_WARMDOWN_RATIO
muon_warmdown_start = 1.0 - WARMDOWN_RATIO
adam_warmdown_start = 1.0 - ADAM_WARMDOWN_RATIO

while True:
    OBS.clear()
    OBS.set_context("train")
    torch.cuda.synchronize()
    t0 = time.time()
    train_accuracy_f = None
    weight_norm_target_reg_observables = {}
    experimental_reg_observables = {}
    for _micro_step in range(grad_accum_steps):
        global_entropy_reg_active = ATTN_ENTROPY_REG and step < ATTN_ENTROPY_REG_UNTIL_STEP
        ATTENTION_ENTROPY_REG_VALUES.clear()
        ATTENTION_ENTROPY_REG_CAPTURE_ACTIVE = global_entropy_reg_active
        for capture in REG_CAPTURES:
            capture.clear()
        with autocast_ctx:
            loss, train_accuracy = model(x, y, return_accuracy=True)
            ATTENTION_ENTROPY_REG_CAPTURE_ACTIVE = False
            task_loss = loss
            if global_entropy_reg_active:
                reg_loss, reg_observables = compute_attention_entropy_regularizer()
                loss = loss + reg_loss
                experimental_reg_observables.update(reg_observables)
            elif ATTN_ENTROPY_REG:
                experimental_reg_observables.update(
                    {
                        "observable_reg_attention_entropy.loss": 0.0,
                        "observable_reg_attention_entropy.layer_count": 0.0,
                    }
                )
            if WEIGHT_NORM_TARGET_REG and step <= WEIGHT_NORM_TARGET_REG_UNTIL_STEP:
                weight_norm_target_reg_loss, weight_norm_target_reg_observables = (
                    compute_weight_norm_target_regularizer(model)
                )
                loss = loss + weight_norm_target_reg_loss
            elif WEIGHT_NORM_TARGET_REG:
                weight_norm_target_reg_observables = {
                    "observable_weight_norm_target_reg.loss": 0.0,
                    "observable_weight_norm_target_reg.target": WEIGHT_NORM_TARGET,
                    "observable_weight_norm_target_reg.num_under_target": 0.0,
                }
            if MLP_L2_GROWTH_REG and step <= MLP_L2_GROWTH_REG_UNTIL_STEP:
                reg_loss, reg_observables = compute_mlp_l2_growth_regularizer(model, step)
                loss = loss + reg_loss
                experimental_reg_observables.update(reg_observables)
            elif MLP_L2_GROWTH_REG:
                experimental_reg_observables.update({"observable_reg_mlp_l2_growth.loss": 0.0})
            if ATTN_V_HEAD_L2_GROWTH_REG and step <= ATTN_HEAD_L2_GROWTH_REG_UNTIL_STEP:
                reg_loss, reg_observables = compute_attention_head_l2_growth_regularizer(
                    model,
                    step,
                    ("c_v",),
                    "attn_v_head_l2_growth",
                )
                loss = loss + reg_loss
                experimental_reg_observables.update(reg_observables)
            elif ATTN_V_HEAD_L2_GROWTH_REG:
                experimental_reg_observables.update({"observable_reg_attn_v_head_l2_growth.loss": 0.0})
            if ATTN_QKV_HEAD_L2_GROWTH_REG and step <= ATTN_HEAD_L2_GROWTH_REG_UNTIL_STEP:
                reg_loss, reg_observables = compute_attention_head_l2_growth_regularizer(
                    model,
                    step,
                    ("c_q", "c_k", "c_v"),
                    "attn_qkv_head_l2_growth",
                )
                loss = loss + reg_loss
                experimental_reg_observables.update(reg_observables)
            elif ATTN_QKV_HEAD_L2_GROWTH_REG:
                experimental_reg_observables.update({"observable_reg_attn_qkv_head_l2_growth.loss": 0.0})
            trajectory_reg_loss, trajectory_reg_observables = compute_trajectory_regularizer(
                model,
                step,
                task_loss,
            )
            loss = loss + trajectory_reg_loss
            experimental_reg_observables.update(trajectory_reg_observables)
        train_accuracy_f = train_accuracy.item()
        train_loss = task_loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules (decoupled warmdown: Muon=0.9, Adam=0.7, Ngram VE=0.0)
    progress = min(step * inv_max_train_steps, 1.0)
    if progress < muon_warmdown_start:
        lrm_muon = 1.0
        muon_wd_frac = 0.0
    else:
        muon_wd_frac = (progress - muon_warmdown_start) * inv_muon_warmdown
        lrm_muon = ((1.0 - progress) * inv_muon_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC

    if progress < adam_warmdown_start:
        lrm_adam = 1.0
        adam_beta1 = ADAM_BETAS[0]
    else:
        adam_wd_frac = (progress - adam_warmdown_start) * inv_adam_warmdown
        lrm_adam = ((1.0 - progress) * inv_adam_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC
        adam_beta1 = ADAM_BETAS[0] + (DEMON_FINAL_BETA1 - ADAM_BETAS[0]) * adam_wd_frac

    frac = min(step / 300, 1)
    muon_momentum = (1 - frac) * 0.85 + frac * MUON_PEAK_MOMENTUM
    if progress > muon_warmdown_start:
        muon_momentum = MUON_PEAK_MOMENTUM + (muon_wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    muon_beta2 = MUON_BETA2_PEAK + muon_wd_frac * (MUON_BETA2_WARMDOWN - MUON_BETA2_PEAK)
    muon_lr_boost = 1.0 + muon_wd_frac * (MUON_LR_BOOST - 1.0)
    # VE RMSProp reverse-Demon: DELAYED ramp (only last 30% of Muon warmdown)
    late_frac = max(0.0, (muon_wd_frac - 0.7) / 0.3)
    ve_beta2 = NGRAM_VE_BETAS[1] + late_frac * (NGRAM_VE_BETA2_WARMDOWN - NGRAM_VE_BETAS[1])

    base_wd = WEIGHT_DECAY * (1 - progress)
    early_dist = abs(progress - WD_EARLY_PULSE_CENTER)
    if early_dist < WD_EARLY_PULSE_HALF_WIDTH:
        muon_weight_decay = base_wd * WD_EARLY_PULSE_MAGNITUDE
    else:
        dist = abs(progress - WD_PULSE_CENTER)
        if dist < WD_PULSE_HALF_WIDTH:
            muon_weight_decay = base_wd * WD_PULSE_MAGNITUDE
        else:
            mid_dist = abs(progress - WD_MID_PULSE_CENTER)
            if mid_dist < WD_MID_PULSE_HALF_WIDTH:
                local = (progress - (WD_MID_PULSE_CENTER - WD_MID_PULSE_HALF_WIDTH)) / (2 * WD_MID_PULSE_HALF_WIDTH)
                bump = 2 * local if local < 0.5 else 2 * (1 - local)
                muon_weight_decay = base_wd * (1.0 + bump * (WD_MID_PULSE_MAGNITUDE - 1.0))
            else:
                muon_weight_decay = base_wd

    muon_lr = lrm_muon * muon_lr_boost
    if progress < muon_warmdown_start:
        for group in muon_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
    else:
        for group, initial_lr in muon_group_lrs:
            group["lr"] = initial_lr * muon_lr
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
        for group, initial_lr in x0_group_lrs:
            group["lr"] = initial_lr * lrm_muon
    if progress >= adam_warmdown_start:
        for group, initial_lr in adam_group_lrs:
            group["lr"] = initial_lr * lrm_adam
        for group, beta2 in adam_demon_groups:
            group["betas"] = (adam_beta1, beta2)
    # Update ngram VE RMSProp beta2 during warmdown (delayed reverse-Demon for sparse tables)
    if progress >= muon_warmdown_start and late_frac > 0.0:
        for group in ngram_groups:
            group["beta2"] = ve_beta2
    hessian_observables = {}
    if OBSERVE_HESSIAN_EIGENVALUES and step % HESSIAN_EVERY == 0:
        hessian_observables = compute_scalar_hessian_observables(model, x, y)
    hessian_observables.update(weight_norm_target_reg_observables)
    gradient_norm_f = compute_gradient_norm(model)
    weight_observables = compute_layer_weight_observables(model)
    weight_observables.update(compute_attention_parameter_weight_observables(model))
    weight_observables.update(compute_mlp_weight_observables(model))
    if OBSERVE_REST100_ATTENTION_HEAD_L2:
        weight_observables.update(compute_rest100_attention_head_l2_observables(model))
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / B200_BF16_PEAK_FLOPS
    remaining = max(0, MAX_TRAIN_STEPS - step - 1)
    val_loss_f = None
    val_accuracy_f = None
    if VAL_LOSS_EVERY > 0 and step % VAL_LOSS_EVERY == 0:
        model.eval()
        OBS.set_context("val")
        with torch.no_grad(), autocast_ctx:
            x_val, y_val, _ = next(val_loader)
            val_loss, val_accuracy = model(x_val, y_val, return_accuracy=True)
            val_loss_f = val_loss.item()
            val_accuracy_f = val_accuracy.item()
        OBS.set_context("train")
        model.train()

    print(
        f"step {step:05d} ({pct_done:.1f}%) | train_loss: {debiased_smooth_loss:.6f} | lrm_muon: {lrm_muon:.2f} lrm_adam: {lrm_adam:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining_steps: {remaining}",
        flush=True,
    )
    if val_loss_f is not None:
        print(f"val_loss: {val_loss_f:.6f} step: {step} epoch: {epoch}", flush=True)
    observables = OBS.flush()
    observables.update(build_step_observables(
        progress=progress,
        train_loss=train_loss_f,
        val_loss=val_loss_f,
        smoothed_train_loss=debiased_smooth_loss,
        train_accuracy=train_accuracy_f,
        val_accuracy=val_accuracy_f,
        gradient_norm=gradient_norm_f,
        weight_observables=weight_observables,
        extra_observables={**hessian_observables, **experimental_reg_observables},
        dt=dt,
        tok_per_sec=tok_per_sec,
        lrm_muon=lrm_muon,
        lrm_adam=lrm_adam,
        muon_momentum=muon_momentum,
        muon_weight_decay=muon_weight_decay,
        mfu=mfu,
    ))
    OBS.record_step(step, epoch, observables)
    if PRINT_OBSERVABLE_LINES:
        print(format_observable_line(step, epoch, observables), flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step >= MAX_TRAIN_STEPS:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100
    * num_flops_per_token
    * TOTAL_BATCH_SIZE
    * (step - 10)
    / total_training_time
    / B200_BF16_PEAK_FLOPS
    if total_training_time > 0
    else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
summary = {
    "val_bpb": val_bpb,
    "training_seconds": total_training_time,
    "total_seconds": t_end - t_start,
    "peak_vram_mb": peak_vram_mb,
    "mfu_percent": steady_state_mfu,
    "total_tokens_M": total_tokens / 1e6,
    "num_steps": step,
    "max_train_steps": MAX_TRAIN_STEPS,
    "num_params_M": num_params / 1e6,
    "depth": DEPTH,
    "seed": _SEED,
    "model_config": asdict(config),
    "observe_layer_probes": OBSERVE_LAYER_PROBES,
    "observe_rest100_attention_head_l2": OBSERVE_REST100_ATTENTION_HEAD_L2,
    "print_observable_lines": PRINT_OBSERVABLE_LINES,
    "weight_norm_target_reg": WEIGHT_NORM_TARGET_REG,
    "weight_norm_target": WEIGHT_NORM_TARGET,
    "weight_norm_target_reg_until_step": WEIGHT_NORM_TARGET_REG_UNTIL_STEP,
    "weight_norm_target_reg_coef": WEIGHT_NORM_TARGET_REG_COEF,
    "mlp_l2_growth_reg": MLP_L2_GROWTH_REG,
    "mlp_l2_growth_reg_layers": list(MLP_L2_GROWTH_REG_LAYERS),
    "mlp_l2_growth_reg_until_step": MLP_L2_GROWTH_REG_UNTIL_STEP,
    "mlp_l2_growth_reg_coef": MLP_L2_GROWTH_REG_COEF,
    "mlp_l2_growth_reg_start_target": MLP_L2_GROWTH_REG_START_TARGET,
    "mlp_l2_growth_reg_end_target": MLP_L2_GROWTH_REG_END_TARGET,
    "attn_v_head_l2_growth_reg": ATTN_V_HEAD_L2_GROWTH_REG,
    "attn_qkv_head_l2_growth_reg": ATTN_QKV_HEAD_L2_GROWTH_REG,
    "attn_head_l2_growth_reg_layers": list(ATTN_HEAD_L2_GROWTH_REG_LAYERS),
    "attn_head_l2_growth_reg_until_step": ATTN_HEAD_L2_GROWTH_REG_UNTIL_STEP,
    "attn_head_l2_growth_reg_coef": ATTN_HEAD_L2_GROWTH_REG_COEF,
    "attn_head_l2_growth_reg_start_target": ATTN_HEAD_L2_GROWTH_REG_START_TARGET,
    "attn_head_l2_growth_reg_end_target": ATTN_HEAD_L2_GROWTH_REG_END_TARGET,
    "attn_entropy_reg": ATTN_ENTROPY_REG,
    "attn_entropy_reg_until_step": ATTN_ENTROPY_REG_UNTIL_STEP,
    "attn_entropy_reg_coef": ATTN_ENTROPY_REG_COEF,
    "trajectory_reg_observable": TRAJECTORY_REG_OBSERVABLE,
    "trajectory_reg_mode": TRAJECTORY_REG_MODE,
    "trajectory_reg_coef": TRAJECTORY_REG_COEF,
    "trajectory_reg_until_step": TRAJECTORY_REG_UNTIL_STEP,
    "trajectory_reg_lead_steps": TRAJECTORY_REG_LEAD_STEPS,
}
observable_csv_path, observable_plot_path, observable_figure_index_path = write_observable_artifacts(summary)

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"observable_csv:   {observable_csv_path}")
print(f"observable_plot:  {observable_plot_path}")
print(f"observable_figures_csv: {observable_figure_index_path}")
