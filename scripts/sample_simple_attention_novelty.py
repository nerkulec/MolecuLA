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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.registry import MODEL_SPECS  # noqa: E402
from models.simple_attention_vae import VaeTransformer  # noqa: E402
from study.common.data import make_splits  # noqa: E402


def canonicalize(smiles: str | None) -> str | None:
    if smiles is None or not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def ids_to_selfies(ids: np.ndarray, id2tok: dict[str, str]) -> str:
    tokens = []
    for value in ids:
        token = id2tok.get(str(int(value)))
        if token is None:
            continue
        if token in {"<PAD>", "<EOS>"}:
            break
        if token == "<SOS>":
            continue
        tokens.append(token)
    return "".join(tokens)


def decode_ids(ids: np.ndarray, id2tok: dict[str, str]) -> tuple[str | None, str | None, str | None, bool]:
    selfies_value = ids_to_selfies(ids, id2tok)
    if not selfies_value:
        return None, None, None, False
    try:
        smiles = sf.decoder(selfies_value)
    except Exception:
        return selfies_value, None, None, False
    canonical = canonicalize(smiles)
    return selfies_value, smiles, canonical, canonical is not None


def load_simple_attention_model(device: torch.device) -> tuple[VaeTransformer, dict, object]:
    spec = MODEL_SPECS["simple_attention"]
    tokenizer_path = REPO_ROOT / "data" / "selfies_tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab_size = int(tokenizer.get("vocab_size", len(tokenizer["vocab"])))

    model = VaeTransformer(
        vocab_size=vocab_size,
        hidden_size=spec.hidden_size,
        latent_size=spec.latent_size,
        max_len=int(tokenizer["max_len"]),
        attn_heads=spec.attention_heads,
        num_slots=spec.num_slots,
        layers=spec.layers,
    )

    try:
        checkpoint = torch.load(str(spec.checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(spec.checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, tokenizer, spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample simple-attention prior latents and measure novelty vs train.")
    parser.add_argument("--n", type=int, default=5000, help="Number of latent prior samples to decode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for NumPy and PyTorch.")
    parser.add_argument("--batch-size", type=int, default=256, help="Decode batch size.")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    start = time.time()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"device={device}")
    print(f"n={args.n}")
    print(f"seed={args.seed}")
    print(f"batch_size={args.batch_size}")

    print("loading dataset and canonicalizing train split...")
    data_path = REPO_ROOT / "data" / "smiles_selfies_full.csv"
    df = pd.read_csv(data_path, usecols=["smiles"])
    splits = make_splits(len(df), seed=args.seed)
    train_smiles = df.iloc[splits["train"]]["smiles"].astype(str)
    train_canonical = {canonical for canonical in (canonicalize(smiles) for smiles in train_smiles) if canonical}

    print(f"dataset_rows={len(df)}")
    print(f"train_rows={len(train_smiles)}")
    print(f"train_unique_canonical={len(train_canonical)}")

    print("loading simple_attention checkpoint...")
    model, tokenizer, spec = load_simple_attention_model(device)
    id2tok = tokenizer["id2tok"]
    print(f"checkpoint={spec.checkpoint}")
    print(f"latent_size={spec.latent_size}")

    valid_canonical: list[str] = []
    pred_lengths: list[float] = []
    preview_rows: list[dict] = []

    print("sampling prior latents and decoding...")
    with torch.no_grad():
        for start_idx in range(0, args.n, args.batch_size):
            stop_idx = min(start_idx + args.batch_size, args.n)
            z = torch.randn(stop_idx - start_idx, spec.latent_size, device=device)
            logits, pred_len = model.decode(z, mode="eval", sos_id=1)
            token_ids = logits.argmax(dim=-1).detach().cpu().numpy()
            pred_lengths.extend(pred_len.detach().cpu().numpy().astype(float).tolist())

            for sample_id, ids in enumerate(token_ids, start=start_idx):
                selfies_value, smiles, canonical, is_valid = decode_ids(ids, id2tok)
                if is_valid and canonical is not None:
                    valid_canonical.append(canonical)
                if len(preview_rows) < 20:
                    preview_rows.append(
                        {
                            "sample_id": sample_id,
                            "is_valid": int(is_valid),
                            "canonical_smiles": canonical,
                            "smiles": smiles,
                            "selfies": selfies_value,
                        }
                    )

    valid_count = len(valid_canonical)
    unique_valid = set(valid_canonical)
    unique_valid_count = len(unique_valid)
    novel_unique = unique_valid - train_canonical
    valid_novel_count = sum(1 for canonical in valid_canonical if canonical not in train_canonical)

    summary = {
        "valid_count": valid_count,
        "validity": valid_count / args.n if args.n else 0.0,
        "unique_valid_count": unique_valid_count,
        "uniqueness_among_valid": unique_valid_count / valid_count if valid_count else 0.0,
        "novel_unique_count": len(novel_unique),
        "novelty_vs_train_unique": len(novel_unique) / unique_valid_count if unique_valid_count else 0.0,
        "novel_valid_sample_count": valid_novel_count,
        "novelty_vs_train_valid_samples": valid_novel_count / valid_count if valid_count else 0.0,
        "train_hits_unique": unique_valid_count - len(novel_unique),
        "train_hits_valid_samples": valid_count - valid_novel_count,
        "pred_len_mean": float(np.mean(pred_lengths)) if pred_lengths else float("nan"),
        "pred_len_min": float(np.min(pred_lengths)) if pred_lengths else float("nan"),
        "pred_len_max": float(np.max(pred_lengths)) if pred_lengths else float("nan"),
        "elapsed_seconds": time.time() - start,
    }

    print("\nSUMMARY")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")

    print("\nFIRST_20_DECODED")
    print(pd.DataFrame(preview_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
