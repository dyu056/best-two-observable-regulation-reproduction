# Best Two-Observable Regulation Reproduction

A self-contained reproduction bundle for the best completed two-observable
trajectory-regulation trial from an exhaustive observable-pair sweep.

## Result

| Metric | Value |
|---|---:|
| Validation BPB | **0.931054588125949** |
| Baseline validation BPB | 0.934416673352 |
| Absolute improvement | **0.003362085226051** |
| Relative improvement | **0.3598%** |
| Sweep row | 404 |
| Seed | 42 |

The winning pair is:

1. `val.layer_2.attn_out.l1`
2. `train.layer_0.k.l1`

Both use normalized trajectory targets with coefficient `0.01`, a 100-step
lead, and regulation through step 750. Training runs for 2,000 steps.

## Repository contents

- `reproduce.py` — isolated, single-command launcher with the winning settings.
- `train.py` — exact historical training source used by the winning sweep trial.
- `curves/` — the two exact target trajectories.
- `summary.json` — authoritative summary from the winning trial.
- `train.log` — authoritative training log from the winning trial.
- `prepare.py`, `lib.py`, `observable.py`, `data_split.json` — data/runtime support.
- `SHA256SUMS.json` — SHA-256 manifest for the published bundle.

## Requirements

- Linux with a CUDA-capable NVIDIA GPU (the recorded run used an NVIDIA H200)
- Python 3.10+
- A PyTorch build compatible with the installed CUDA driver

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For production GPU environments, install the appropriate PyTorch wheel for the
machine before installing the remaining requirements.

## Prepare data

The training data and tokenizer are stored under `~/.cache/autoresearch/`:

```bash
python prepare.py
```

`prepare.py` downloads the shard split specified by `data_split.json` and builds
the tokenizer.

## Reproduce

Run on a selected physical GPU, for example GPU 0:

```bash
python reproduce.py \
  --python "$(which python)" \
  --gpu 0 \
  --output-dir rerun
```

The run writes:

- `rerun/train.log`
- `rerun/summary.json`
- isolated working files under `rerun/work/`

Compare `rerun/summary.json` with the published `summary.json`. Exact bitwise
identity can depend on GPU model, CUDA/PyTorch versions, and kernel selection;
the repository preserves the exact source, seed, curves, and hyperparameters.

## Winning configuration

```text
observable_1 = val.layer_2.attn_out.l1
observable_2 = train.layer_0.k.l1
mode_1 = trajectory
mode_2 = trajectory
coefficient_1 = 0.01
coefficient_2 = 0.01
lead_steps = 100
until_step = 750
seed = 42
max_train_steps = 2000
```

## Integrity

Verify the published files:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('.')
manifest = json.loads((root / 'SHA256SUMS.json').read_text())
for name, expected in manifest.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert actual == expected, f'{name}: {actual} != {expected}'
print(f'Verified {len(manifest)} files')
PY
```

## License

Apache License 2.0. See `LICENSE`.
