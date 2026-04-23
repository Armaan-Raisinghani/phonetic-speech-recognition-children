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
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PhonemeDataset, ctc_collate
from src.decode import greedy_ctc_decode
from src.metrics import cer, per
from src.model import PhonemeCTCModel
from src.text import decode_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    return p.parse_args()


def load_vocab(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in payload["stoi"].items()}
    stoi = {k: int(v) for k, v in stoi.items()}
    itos = {int(k): v for k, v in payload["itos"].items()}
    return stoi, itos


def batch_ctc_loss(
    logits: torch.Tensor,
    out_lens: torch.Tensor,
    targets: torch.Tensor,
    target_lens: torch.Tensor,
    label_weights: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
    ctc = torch.nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=True)
    loss_vec = ctc(log_probs, targets, out_lens, target_lens)
    norm_w = label_weights / torch.clamp(label_weights.sum(), min=1e-6)
    return (loss_vec * norm_w).sum()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    itos: dict[int, str],
    blank_id: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_per = []
    all_cer = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            feat_lens = batch["feature_lens"].to(device)
            logits, _ = model(feats, feat_lens)
            pred_ids = greedy_ctc_decode(logits, blank_id=blank_id)
            pred_texts = [decode_ids(ids, itos) for ids in pred_ids]
            for ref, hyp in zip(batch["texts"], pred_texts):
                all_per.append(per(ref, hyp))
                all_cer.append(cer(ref, hyp))

    if not all_per:
        return {"per": 1.0, "cer": 1.0}
    return {
        "per": float(sum(all_per) / len(all_per)),
        "cer": float(sum(all_cer) / len(all_cer)),
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stoi, itos = load_vocab(Path(cfg["data"]["vocab_path"]))
    blank_id = stoi["<blank>"]

    train_ds = PhonemeDataset(
        manifest_path=Path(cfg["data"]["train_manifest"]),
        stoi=stoi,
        train=True,
        use_specaugment=cfg["data"].get("use_specaugment", True),
    )
    val_ds = PhonemeDataset(
        manifest_path=Path(cfg["data"]["val_manifest"]),
        stoi=stoi,
        train=False,
        use_specaugment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=ctc_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=ctc_collate,
    )

    model = PhonemeCTCModel(
        vocab_size=len(stoi),
        hidden_dim=cfg["model"].get("hidden_dim", 512),
        freeze_backbone=cfg["model"].get("freeze_backbone", True),
        use_transfer=cfg["model"].get("use_transfer", True),
    ).to(device)
    print(json.dumps({"transfer_loaded": bool(model.transfer_loaded)}))

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_per = 1e9
    out_dir = Path(cfg["train"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    unfreeze_epoch = int(cfg["train"].get("unfreeze_epoch", 3))
    grad_clip = float(cfg["train"].get("grad_clip", 5.0))

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        if epoch == unfreeze_epoch:
            model.unfreeze_backbone()
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

        model.train()
        running_loss = 0.0

        prog = tqdm(train_loader, desc=f"epoch {epoch}", ncols=100)
        for batch in prog:
            feats = batch["features"].to(device)
            feat_lens = batch["feature_lens"].to(device)
            targets = batch["targets"].to(device)
            target_lens = batch["target_lens"].to(device)
            label_weights = batch["label_weights"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, out_lens = model(feats, feat_lens)
                loss = batch_ctc_loss(
                    logits,
                    out_lens,
                    targets,
                    target_lens,
                    label_weights,
                    blank_id,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.item())
            prog.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, itos, blank_id, device)
        avg_loss = running_loss / max(1, len(train_loader))
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_per": val_metrics["per"],
                    "val_cer": val_metrics["cer"],
                }
            )
        )

        ckpt = {
            "model_state": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "val_per": val_metrics["per"],
        }
        torch.save(ckpt, out_dir / "last.pt")

        if val_metrics["per"] < best_per:
            best_per = val_metrics["per"]
            torch.save(ckpt, out_dir / "best.pt")


if __name__ == "__main__":
    main()
