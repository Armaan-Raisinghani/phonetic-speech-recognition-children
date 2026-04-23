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
from src.decode import greedy_ctc_decode
from src.metrics import cer, per
from src.model import PhonemeCTCModel
from src.text import decode_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("results/eval_predictions.jsonl"))
    return p.parse_args()


def load_vocab(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in payload["stoi"].items()}
    stoi = {k: int(v) for k, v in stoi.items()}
    itos = {int(k): v for k, v in payload["itos"].items()}
    return stoi, itos


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stoi, itos = load_vocab(args.vocab)
    blank_id = stoi["<blank>"]

    ds = PhonemeDataset(args.manifest, stoi=stoi, train=False, use_specaugment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    hidden_dim = cfg.get("model", {}).get("hidden_dim", 512)

    model = PhonemeCTCModel(vocab_size=len(stoi), hidden_dim=hidden_dim, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_vals = []
    cer_vals = []

    with args.out.open("w", encoding="utf-8") as fw, torch.no_grad():
        for batch in dl:
            feats = batch["features"].to(device)
            feat_lens = batch["feature_lens"].to(device)
            logits, _ = model(feats, feat_lens)
            pred_ids = greedy_ctc_decode(logits, blank_id=blank_id)
            pred_texts = [decode_ids(ids, itos) for ids in pred_ids]

            for utt, ref, hyp in zip(batch["utt_ids"], batch["texts"], pred_texts):
                p = per(ref, hyp)
                c = cer(ref, hyp)
                per_vals.append(p)
                cer_vals.append(c)
                fw.write(
                    json.dumps(
                        {
                            "utterance_id": utt,
                            "reference": ref,
                            "prediction": hyp,
                            "per": p,
                            "cer": c,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "manifest": str(args.manifest),
        "num_samples": len(per_vals),
        "avg_per": float(sum(per_vals) / max(1, len(per_vals))),
        "avg_cer": float(sum(cer_vals) / max(1, len(cer_vals))),
        "predictions": str(args.out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
