#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

from lignin_pipeline.common import REPO_ROOT, ids_to_selfies, ids_until_pad, json_dump, percentage, read_rows, write_rows


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.registry import MODEL_SPECS  # noqa: E402


MODEL_NAMES = ("linear_attention", "simple_attention", "autoregressive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode lignin SELFIES, save mu latents, and evaluate reconstruction.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--eligible-mask", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mismatch-limit", type=int, default=100)
    parser.add_argument("--torch-threads", type=int)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(name: str, vocab_size: int, device: torch.device):
    spec = MODEL_SPECS[name]
    if name == "linear_attention":
        from models.linear_attention_vae import VaeTransformer

        model = VaeTransformer(
            vocab_size=vocab_size,
            hidden_size=spec.hidden_size,
            latent_size=spec.latent_size,
            max_len=spec.max_len,
            attn_heads=spec.attention_heads,
            num_slots=spec.num_slots,
            layers=spec.layers,
        )
    elif name == "simple_attention":
        from models.simple_attention_vae import VaeTransformer

        model = VaeTransformer(
            vocab_size=vocab_size,
            hidden_size=spec.hidden_size,
            latent_size=spec.latent_size,
            max_len=spec.max_len,
            attn_heads=spec.attention_heads,
            num_slots=spec.num_slots,
            layers=spec.layers,
        )
    else:
        from models.autoregressive_vae import VaeTransformer

        model = VaeTransformer(
            vocab_size=vocab_size,
            hidden_size=spec.hidden_size,
            latent_size=spec.latent_size,
            max_len=spec.max_len,
            attn_heads=spec.attention_heads,
            num_slots=spec.num_slots,
            encoder_layers=spec.encoder_layers,
            decoder_layers=spec.decoder_layers,
        )
    try:
        checkpoint = torch.load(str(spec.checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(spec.checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    embedding = state.get("embedding.weight")
    if embedding is None or embedding.shape[0] != vocab_size:
        raise ValueError(f"Checkpoint/tokenizer vocabulary mismatch: {None if embedding is None else embedding.shape[0]} != {vocab_size}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), spec


def canonical_smiles(smiles_value: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles_value, sanitize=True)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else None
    except Exception:
        return None


def trim_ar_prediction(row: np.ndarray) -> tuple[np.ndarray, bool]:
    eos = np.flatnonzero(row == 2)
    if len(eos):
        return row[: int(eos[0]) + 1], True
    return row, False


def pad_for_comparison(row: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.int64)
    used = min(length, len(row))
    out[:used] = row[:used]
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    rows = read_rows(args.rows)
    all_tokens = np.load(args.tokens, mmap_mode="r", allow_pickle=False)
    eligible_mask = np.load(args.eligible_mask, allow_pickle=False).astype(bool)
    tokenizer = json.loads(args.tokenizer.read_text(encoding="utf-8"))
    if len(rows) != len(all_tokens) or len(rows) != len(eligible_mask):
        raise ValueError("Rows, tokens, and eligibility mask are not aligned")
    eligible_indices = np.flatnonzero(eligible_mask)
    if not len(eligible_indices):
        raise ValueError("No eligible rows")
    model, spec = build_model(args.model, len(tokenizer["vocab"]), device)
    id2tok = tokenizer["id2tok"]
    started = time.time()
    latent_chunks: list[np.ndarray] = []
    result_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    token_correct = 0
    token_total = 0

    with torch.inference_mode():
        batches = range(0, len(eligible_indices), args.batch_size)
        for start in tqdm(batches, desc=f"{args.model} encode/decode", unit="batch"):
            batch_indices = eligible_indices[start : start + args.batch_size]
            token_np = np.asarray(all_tokens[batch_indices], dtype=np.int64)
            x = torch.as_tensor(token_np, dtype=torch.long, device=device)
            mu, _ = model.encode(x)
            latent_chunks.append(mu.detach().cpu().numpy().astype(np.float32, copy=False))

            if args.model == "autoregressive":
                decoded_tensor = model.decode(mu, x_in=None)
                decoded = decoded_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
                predicted_lengths = None
                raw_logits_ids = None
            else:
                logits, predicted_length_tensor = model.decode(mu, mode="eval", sos_id=1)
                raw_logits_ids = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)
                predicted_lengths = np.rint(predicted_length_tensor.detach().cpu().numpy()).astype(np.int64)
                predicted_lengths = np.maximum(predicted_lengths, 1)
                decoded = None

            for local_index, source_index in enumerate(batch_indices):
                target = ids_until_pad(token_np[local_index]).astype(np.int64, copy=False)
                if args.model == "autoregressive":
                    prediction, emitted_eos = trim_ar_prediction(decoded[local_index])
                    compatible_source = prediction
                else:
                    emitted_eos = bool(2 in raw_logits_ids[local_index, : predicted_lengths[local_index]])
                    prediction = raw_logits_ids[local_index, : predicted_lengths[local_index]]
                    compatible_source = raw_logits_ids[local_index]

                strict_exact = len(prediction) == len(target) and bool(np.array_equal(prediction, target))
                compatible = pad_for_comparison(compatible_source, len(target))
                compatibility_exact = bool(np.array_equal(compatible, target))
                strict_for_tokens = pad_for_comparison(prediction, len(target))
                token_correct += int((strict_for_tokens == target).sum())
                token_total += len(target)
                predicted_selfies, special_error = ids_to_selfies(prediction, id2tok)
                selfies_valid = False
                smiles_valid = False
                equivalent = False
                predicted_smiles = None
                predicted_canonical = None
                if special_error is None and predicted_selfies:
                    try:
                        predicted_smiles = sf.decoder(predicted_selfies)
                        selfies_valid = True
                    except Exception:
                        predicted_smiles = None
                if predicted_smiles:
                    predicted_canonical = canonical_smiles(predicted_smiles)
                    smiles_valid = predicted_canonical is not None
                    equivalent = smiles_valid and predicted_canonical == str(rows.iloc[source_index]["rdkit_smiles"])
                record = {
                    "rowid": int(rows.iloc[source_index]["rowid"]),
                    "true_length": len(target),
                    "predicted_length": len(prediction),
                    "strict_exact": int(strict_exact),
                    "repository_compatible_exact": int(compatibility_exact),
                    "length_exact": int(len(prediction) == len(target)),
                    "emitted_eos": int(emitted_eos),
                    "selfies_valid": int(selfies_valid),
                    "smiles_valid": int(smiles_valid),
                    "canonical_equivalent": int(equivalent),
                    "special_token_error": special_error or "",
                }
                result_rows.append(record)
                if not strict_exact and len(mismatch_rows) < args.mismatch_limit:
                    mismatch_rows.append(
                        {
                            **record,
                            "target_selfies": rows.iloc[source_index]["selfies_final"],
                            "predicted_selfies": predicted_selfies or "",
                            "target_smiles": rows.iloc[source_index]["rdkit_smiles"],
                            "predicted_smiles": predicted_smiles or "",
                        }
                    )

    latents = np.concatenate(latent_chunks, axis=0)
    rowids = rows.iloc[eligible_indices]["rowid"].to_numpy(dtype=np.int64)
    np.save(args.output_dir / "latents.npy", latents, allow_pickle=False)
    np.save(args.output_dir / "rowids.npy", rowids, allow_pickle=False)
    result_frame = pd.DataFrame(result_rows)
    write_rows(result_frame, args.output_dir / "reconstruction_rows.csv.gz")
    write_rows(pd.DataFrame(mismatch_rows), args.output_dir / "reconstruction_mismatches.csv.gz")

    n = len(result_frame)
    strict_failures = int((result_frame["strict_exact"] == 0).sum())
    length_failures = int((result_frame["length_exact"] == 0).sum())
    eos_failures = int((result_frame["emitted_eos"] == 0).sum())
    selfies_failures = int((result_frame["selfies_valid"] == 0).sum())
    smiles_failures = int((result_frame["smiles_valid"] == 0).sum())
    canonical_failures = int((result_frame["canonical_equivalent"] == 0).sum())
    report = {
        "stage": "encode_decode",
        "model": args.model,
        "checkpoint": str(spec.checkpoint),
        "device": str(device),
        "batch_size": args.batch_size,
        "rows": n,
        "latent_dim": int(latents.shape[1]),
        "token_accuracy_percent": percentage(token_correct, token_total),
        "token_correct": token_correct,
        "token_total": token_total,
        "strict_exact_percent": percentage(int(result_frame["strict_exact"].sum()), n),
        "strict_reconstruction_failure_count": strict_failures,
        "strict_reconstruction_failure_percent": percentage(strict_failures, n),
        "repository_compatible_exact_percent": percentage(int(result_frame["repository_compatible_exact"].sum()), n),
        "length_failure_count": length_failures,
        "length_failure_percent": percentage(length_failures, n),
        "eos_failure_count": eos_failures,
        "eos_failure_percent": percentage(eos_failures, n),
        "selfies_decode_failure_count": selfies_failures,
        "selfies_decode_failure_percent": percentage(selfies_failures, n),
        "smiles_validation_failure_count": smiles_failures,
        "smiles_validation_failure_percent": percentage(smiles_failures, n),
        "canonical_mismatch_count": canonical_failures,
        "canonical_mismatch_percent": percentage(canonical_failures, n),
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
    }
    json_dump(report, args.output_dir / "reconstruction_report.json")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
