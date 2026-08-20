"""Memory-mapped packed dataset and deterministic length-bucketed sampler."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


class PackedLigninDataset(Dataset):
    def __init__(self, shard_dirs: list[Path], split: str, limit: int | None = None):
        self.shards = []
        shard_id_parts = []
        local_id_parts = []
        length_parts = []
        split_id = SPLIT_IDS[split]
        tokenizer_hashes = set()
        for shard_id, directory in enumerate(sorted(shard_dirs)):
            manifest = json.loads((directory / "manifest.json").read_text())
            tokenizer_hashes.add(manifest["tokenizer_sha256"])
            shard = {
                "tokens": np.load(directory / "tokens.npy", mmap_mode="r"),
                "offsets": np.load(directory / "offsets.npy", mmap_mode="r"),
                "lengths": np.load(directory / "lengths.npy", mmap_mode="r"),
                "rowids": np.load(directory / "rowids.npy", mmap_mode="r"),
            }
            self.shards.append(shard)
            local = np.flatnonzero(np.load(directory / "splits.npy", mmap_mode="r") == split_id)
            shard_id_parts.append(np.full(len(local), shard_id, dtype=np.uint16))
            local_id_parts.append(local.astype(np.int32, copy=False))
            length_parts.append(np.asarray(shard["lengths"][local], dtype=np.int32))
        if len(tokenizer_hashes) != 1:
            raise ValueError(f"Encoded shards use different tokenizers: {tokenizer_hashes}")
        self.tokenizer_sha256 = next(iter(tokenizer_hashes))
        self.shard_ids = np.concatenate(shard_id_parts) if shard_id_parts else np.empty(0, dtype=np.uint16)
        self.local_ids = np.concatenate(local_id_parts) if local_id_parts else np.empty(0, dtype=np.int32)
        self.lengths = np.concatenate(length_parts) if length_parts else np.empty(0, dtype=np.int32)
        if limit is not None:
            self.shard_ids = self.shard_ids[:limit]
            self.local_ids = self.local_ids[:limit]
            self.lengths = self.lengths[:limit]

    def __len__(self):
        return len(self.local_ids)

    def __getitem__(self, index):
        shard_id, local = int(self.shard_ids[index]), int(self.local_ids[index])
        shard = self.shards[shard_id]
        start, stop = int(shard["offsets"][local]), int(shard["offsets"][local + 1])
        return torch.as_tensor(np.asarray(shard["tokens"][start:stop], dtype=np.int64)), int(shard["rowids"][local])


def pad_collate(batch):
    sequences, rowids = zip(*batch)
    result = torch.zeros((len(sequences), max(map(len, sequences))), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        result[i, : len(sequence)] = sequence
    return result, torch.tensor(rowids, dtype=torch.long)


class BucketBatchSampler(Sampler[list[int]]):
    """Shuffle large buckets, sort within each bucket, then distribute full batches."""
    def __init__(self, lengths, batch_size, shuffle, seed=42, rank=0, world_size=1, bucket_multiple=100):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.bucket_size = batch_size * bucket_multiple
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        total_batches = math.ceil(len(self.lengths) / self.batch_size)
        return math.ceil(total_batches / self.world_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indexes = np.arange(len(self.lengths))
        if self.shuffle:
            rng.shuffle(indexes)
        batches = []
        for start in range(0, len(indexes), self.bucket_size):
            bucket = indexes[start : start + self.bucket_size]
            bucket = bucket[np.argsort(self.lengths[bucket], kind="stable")]
            batches.extend(bucket[i : i + self.batch_size].tolist() for i in range(0, len(bucket), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        # Pad the batch list across ranks by repeating deterministic early batches.
        target = math.ceil(len(batches) / self.world_size) * self.world_size
        batches.extend(batches[: target - len(batches)])
        yield from batches[self.rank:target:self.world_size]
