#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.dataset import PhonemeDataset, ctc_collate
from src.decode import confidence_from_logits, greedy_ctc_decode
from src.model import PhonemeCTCModel
from src.text import decode_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--unlabeled-manifest", type=Path, required=True)
    p.add_argument("--supervised-train-manifest", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--keep-ratio", type=float, default=0.2)
    p.add_argument("--pseudo-weight", type=float, default=0.4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-pseudo", type=Path, default=Path("data/manifests/pseudo_labels.jsonl"))
    p.add_argument(
        "--out-merged-train",
        type=Path,
        default=Path("data/manifests/train_with_pseudo.jsonl"),
    )
    return p.parse_args()


def load_vocab(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in payload["stoi"].items()}
    stoi = {k: int(v) for k, v in stoi.items()}
    itos = {int(k): v for k, v in payload["itos"].items()}
    return stoi, itos


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stoi, itos = load_vocab(args.vocab)
    blank_id = stoi["<blank>"]

    ds = PhonemeDataset(args.unlabeled_manifest, stoi=stoi, train=False, use_specaugment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    hidden_dim = cfg.get("model", {}).get("hidden_dim", 512)

    model = PhonemeCTCModel(vocab_size=len(stoi), hidden_dim=hidden_dim, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    candidates: list[dict] = []

    with torch.no_grad():
        for batch in dl:
            feats = batch["features"].to(device)
            feat_lens = batch["feature_lens"].to(device)
            logits, _ = model(feats, feat_lens)
            conf = confidence_from_logits(logits).cpu().tolist()
            pred_ids = greedy_ctc_decode(logits, blank_id=blank_id)
            pred_texts = [decode_ids(ids, itos) for ids in pred_ids]

            for utt_id, text, score in zip(batch["utt_ids"], pred_texts, conf):
                if not text:
                    continue
                candidates.append(
                    {
                        "utterance_id": utt_id,
                        "prediction": text,
                        "confidence": float(score),
                    }
                )

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    keep_n = max(1, int(len(candidates) * args.keep_ratio))
    kept = candidates[:keep_n]

    unlabeled_rows = {r["utterance_id"]: r for r in read_jsonl(args.unlabeled_manifest)}
    pseudo_rows = []
    for item in kept:
        base = unlabeled_rows[item["utterance_id"]]
        pseudo_rows.append(
            {
                "utterance_id": base["utterance_id"],
                "child_id": "pseudo",
                "session_id": "pseudo",
                "audio_path": base["audio_path"],
                "audio_duration_sec": 0.0,
                "age_bucket": "unknown",
                "phonetic_text": item["prediction"],
                "split": "train",
                "is_pseudo": True,
                "label_weight": float(args.pseudo_weight),
                "confidence": item["confidence"],
            }
        )

    sup_train_rows = read_jsonl(args.supervised_train_manifest)
    merged = sup_train_rows + pseudo_rows

    write_jsonl(args.out_pseudo, pseudo_rows)
    write_jsonl(args.out_merged_train, merged)

    print(
        json.dumps(
            {
                "num_candidates": len(candidates),
                "keep_ratio": args.keep_ratio,
                "num_pseudo_kept": len(pseudo_rows),
                "out_pseudo": str(args.out_pseudo),
                "out_merged_train": str(args.out_merged_train),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
