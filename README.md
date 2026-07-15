# Unregularized Baseline — Comparison Branch

This branch contains the **baseline code and recorded baseline result** used to
measure the improvement published on the repository's [`main`](../../tree/main)
branch.

Both the baseline and winning pair were evaluated on a single **NVIDIA H200
GPU**, using 2,000 training steps and seed 42.

## Baseline result

| Metric | Value |
|---|---:|
| Baseline validation BPB | **0.9344166733521407** |
| Winning pair validation BPB | 0.931054588125949 |
| Improvement from baseline | **0.0033620852261917 BPB** |
| Relative improvement | **0.3598%** |
| Steps | 2,000 |
| Seed | 42 |

## What differs from `main`

The baseline `train.py` is the historical pre-regulation source from parent
project commit `4bc2743a` (`Add observable probes and result plots`). It contains
no trajectory-regulation implementation. The `main` branch adds two-observable
trajectory regulation and supplies the winning target curves:

- `val.layer_2.attn_out.l1`
- `train.layer_0.k.l1`

This makes the GitHub branch comparison a direct code-level view of the baseline
versus the published solution.

## Repository contents

- `train.py` — exact historical baseline training source.
- `reproduce.py` — isolated launcher for the unregularized baseline.
- `summary.json` — authoritative baseline result.
- `prepare.py`, `lib.py`, `observable.py`, `data_split.json` — data/runtime support.
- `SHA256SUMS.json` — integrity manifest for this branch.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py
```

The data and tokenizer are stored under `~/.cache/autoresearch/`.

## Run the baseline

```bash
python reproduce.py \
  --python "$(which python)" \
  --gpu 0 \
  --output-dir rerun
```

Outputs are written to `rerun/train.log`, `rerun/summary.json`, and the isolated
`rerun/work/` directory. Observable CSVs and plots are also generated because
the recorded baseline had the layer and attention-head probes enabled.

## Compare with the winning solution

- [Baseline branch](../../tree/baseline)
- [Winning solution (`main`)](../../tree/main)
- [Full branch diff](../../compare/baseline...main)

## Source identity

The baseline `train.py` SHA-256 is:

```text
abf1107b09e2672d2439bceb8bf75d1d7a3a8ee7b4c4e1502d2bc21a43485dd2
```

## License

Apache License 2.0. See `LICENSE`.
