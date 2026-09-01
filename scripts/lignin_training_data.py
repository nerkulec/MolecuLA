"""Memory-mapped packed dataset and deterministic length-bucketed sampler."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm


SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


class PackedLigninDataset(Dataset):
    def __init__(self, shard_dirs: list[Path], split: str | None, limit: int | None = None):
        self.shards = []
        shard_id_parts = []
        local_id_parts = []
        length_parts = []
        split_id = SPLIT_IDS[split] if split is not None else None
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
            if split_id is None:
                local = np.arange(len(shard["lengths"]), dtype=np.int64)
            else:
                local = np.flatnonzero(
                    np.load(directory / "splits.npy", mmap_mode="r") == split_id
                )
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


class GpuPackedLigninDataset:
    """One packed GPU token buffer shared by train/validation split views."""

    def __init__(self, shard_dirs: list[Path], device: torch.device):
        directories = sorted(shard_dirs)
        manifests = [json.loads((directory / "manifest.json").read_text()) for directory in directories]
        tokenizer_hashes = {manifest["tokenizer_sha256"] for manifest in manifests}
        if len(tokenizer_hashes) != 1:
            raise ValueError(f"Encoded shards use different tokenizers: {tokenizer_hashes}")
        self.tokenizer_sha256 = next(iter(tokenizer_hashes))
        total_rows = sum(int(manifest["rows"]) for manifest in manifests)
        total_tokens = sum(int(manifest["tokens"]) for manifest in manifests)
        estimated_bytes = total_tokens * 4 + (total_rows + 1) * 8
        print(
            f"Loading {total_rows:,} sequences and {total_tokens:,} packed tokens "
            f"onto {device} ({estimated_bytes / 2**30:.2f} GiB)",
            flush=True,
        )
        self.tokens = torch.empty(total_tokens, dtype=torch.int32, device=device)
        self.offsets = torch.empty(total_rows + 1, dtype=torch.int64, device=device)
        self.offsets[0] = 0
        lengths_parts: list[np.ndarray] = []
        split_parts: list[np.ndarray] = []
        row_cursor = token_cursor = 0
        progress = tqdm(
            zip(directories, manifests, strict=True),
            total=len(directories),
            desc="load dataset to GPU",
            unit="shard",
            dynamic_ncols=True,
        )
        for directory, manifest in progress:
            tokens = np.load(directory / "tokens.npy", mmap_mode="r")
            offsets = np.load(directory / "offsets.npy", mmap_mode="r")
            lengths = np.load(directory / "lengths.npy", mmap_mode="r")
            splits = np.load(directory / "splits.npy", mmap_mode="r")
            rows = int(manifest["rows"])
            token_count = int(manifest["tokens"])
            self.tokens[token_cursor : token_cursor + token_count].copy_(
                torch.as_tensor(np.asarray(tokens, dtype=np.int32), device=device)
            )
            adjusted_offsets = np.asarray(offsets[1:], dtype=np.int64) + token_cursor
            self.offsets[row_cursor + 1 : row_cursor + rows + 1].copy_(
                torch.as_tensor(adjusted_offsets, device=device)
            )
            lengths_parts.append(np.asarray(lengths, dtype=np.int32))
            split_parts.append(np.asarray(splits, dtype=np.uint8))
            row_cursor += rows
            token_cursor += token_count
            progress.set_postfix_str(f"{token_cursor / 1e9:.2f}B tokens")
        self.lengths = np.concatenate(lengths_parts)
        self.splits = np.concatenate(split_parts)
        self.device = device

    def split(self, name: str, limit: int | None = None):
        indexes = np.flatnonzero(self.splits == SPLIT_IDS[name])
        if limit is not None:
            indexes = indexes[:limit]
        return GpuPackedSplit(self, indexes)

    def all(self, limit: int | None = None):
        indexes = np.arange(len(self.lengths), dtype=np.int64)
        if limit is not None:
            indexes = indexes[:limit]
        return GpuPackedSplit(self, indexes)


class GpuPackedSplit:
    def __init__(self, parent: GpuPackedLigninDataset, indexes: np.ndarray):
        self.parent = parent
        self.indexes = indexes.astype(np.int64, copy=False)
        self.lengths = parent.lengths[self.indexes]
        self.tokenizer_sha256 = parent.tokenizer_sha256

    def __len__(self):
        return len(self.indexes)

    def fetch(self, batch_indexes: list[int]):
        global_indexes = self.indexes[np.asarray(batch_indexes, dtype=np.int64)]
        indexes = torch.as_tensor(global_indexes, dtype=torch.int64, device=self.parent.device)
        starts = self.parent.offsets.index_select(0, indexes)
        stops = self.parent.offsets.index_select(0, indexes + 1)
        lengths = stops - starts
        width = int(lengths.max().item())
        positions = torch.arange(width, device=self.parent.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        packed_indexes = starts.unsqueeze(1) + positions.unsqueeze(0)
        packed_indexes.masked_fill_(~valid, 0)
        batch = self.parent.tokens[packed_indexes]
        batch.masked_fill_(~valid, 0)
        return batch, indexes + 1


class GpuBatchLoader:
    def __init__(self, dataset: GpuPackedSplit, batch_sampler: Sampler[list[int]]):
        self.dataset = dataset
        self.batch_sampler = batch_sampler

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for indexes in self.batch_sampler:
            yield self.dataset.fetch(indexes)


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
