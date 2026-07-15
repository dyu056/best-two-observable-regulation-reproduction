# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
"""Runtime utilities: tokenizer wrapper, dataloader, and BPB evaluation."""

import os
import json
import math
import pickle

import pyarrow.parquet as pq
import torch

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288

# H200 adapter: Recursive's upstream baseline expects /data, while our prepared
# Karpathy shards/tokenizer live under ~/.cache/autoresearch when /data is absent.
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
_USE_UPSTREAM_DATA_DIR = os.path.isdir("/data")
DATA_DIR = "/data" if _USE_UPSTREAM_DATA_DIR else os.path.join(_CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer") if _USE_UPSTREAM_DATA_DIR else os.path.join(_CACHE_DIR, "tokenizer")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_SPLIT_PATH = os.path.join(PROJECT_ROOT, "data_split.json")
BOS_TOKEN = "<|reserved_0|>"


def _shard_filename(index):
    return f"shard_{index:05d}.parquet"


def _normalize_shard_ids(values, split_name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"data_split.json must define non-empty list '{split_name}'")
    seen = set()
    ids = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid shard id in '{split_name}': {value!r}")
        if value in seen:
            raise ValueError(f"Duplicate shard id in '{split_name}': {value}")
        seen.add(value)
        ids.append(value)
    return ids


def load_data_split():
    with open(DATA_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_ids = _normalize_shard_ids(split.get("train"), "train")
    test_ids = _normalize_shard_ids(split.get("test"), "test")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"Shard ids cannot appear in both train and test: {overlap}")
    return {"train": train_ids, "test": test_ids}


def _canonical_split(split):
    if split == "train":
        return "train"
    if split in ("val", "test"):
        return "test"
    raise ValueError(f"Unknown split: {split!r}")


def split_parquet_files(split):
    split_ids = load_data_split()[_canonical_split(split)]
    return [os.path.join(DATA_DIR, _shard_filename(index)) for index in split_ids]


def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    parquet_paths = split_parquet_files(split)
    missing = [p for p in parquet_paths if not os.path.exists(p)]
    assert not missing, f"Missing {split} shards. Run prepare.py first: {missing}"
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """BOS-aligned dataloader with best-fit packing. Every row starts with BOS;
    documents are packed best-fit, cropping the shortest doc when nothing fits."""
    assert split in ["train", "val", "test"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """Bits per byte: vocab-size-independent metric. Sums per-token
    cross-entropy (nats) and target byte lengths, converts nats/byte to
    bits/byte; special tokens (byte length 0) are excluded."""
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
