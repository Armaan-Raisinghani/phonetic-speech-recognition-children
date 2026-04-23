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

from src.decode import decode_logits
from src.features import LogMelFrontend, load_audio
from src.model import PhonemeCTCModel
from src.text import decode_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--decode-strategy", type=str, default=None)
    p.add_argument("--beam-size", type=int, default=None)
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

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    hidden_dim = cfg.get("model", {}).get("hidden_dim", 512)
    decode_cfg = cfg.get("decode", {})
    decode_strategy = (args.decode_strategy or decode_cfg.get("strategy") or "beam").strip().lower()
    beam_size = max(1, int(args.beam_size or decode_cfg.get("beam_size", 8)))

    model = PhonemeCTCModel(vocab_size=len(stoi), hidden_dim=hidden_dim, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    frontend = LogMelFrontend(sample_rate=16000)

    waveform, _ = load_audio(str(args.audio), target_sr=16000)
    feat = frontend(waveform).unsqueeze(0).to(device)
    feat_lens = torch.tensor([feat.size(1)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(feat, feat_lens)
        pred_ids = decode_logits(
            logits,
            blank_id=blank_id,
            strategy=decode_strategy,
            beam_size=beam_size,
        )[0]
        pred_text = decode_ids(pred_ids, itos)

    print(pred_text)


if __name__ == "__main__":
    main()
