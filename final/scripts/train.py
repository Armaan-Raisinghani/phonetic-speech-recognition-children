#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader, Sampler
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


def resolve_num_workers(cfg: dict) -> int:
    override = os.environ.get("NUM_WORKERS")
    if override is not None:
        return int(override)
    return int(cfg["train"].get("num_workers", 4))


def resolve_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "false"}:
        return None
    return int(value)


class DurationBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        durations: list[float],
        batch_size: int,
        shuffle: bool = True,
        bucket_size_multiplier: int = 20,
    ) -> None:
        self.durations = durations
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.bucket_size = max(self.batch_size, self.batch_size * int(bucket_size_multiplier))

    def __iter__(self):
        indices = list(range(len(self.durations)))
        if self.shuffle:
            random.shuffle(indices)
            pooled: list[int] = []
            for start in range(0, len(indices), self.bucket_size):
                pool = indices[start : start + self.bucket_size]
                pool.sort(key=self.durations.__getitem__)
                pooled.extend(pool)
            indices = pooled
        else:
            indices.sort(key=self.durations.__getitem__)

        batches = [indices[start : start + self.batch_size] for start in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.durations) + self.batch_size - 1) // self.batch_size


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

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batch_size = int(cfg["train"]["batch_size"])
    eval_batch_size = int(cfg["train"].get("eval_batch_size", train_batch_size))
    grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    num_workers = resolve_num_workers(cfg)
    freeze_backbone = bool(cfg["model"].get("freeze_backbone", True))
    unfreeze_epoch = resolve_optional_int(cfg["train"].get("unfreeze_epoch"))

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

    train_batch_sampler = DurationBucketBatchSampler(
        durations=[float(row.get("audio_duration_sec", 0.0)) for row in train_ds.rows],
        batch_size=train_batch_size,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        collate_fn=ctc_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ctc_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = PhonemeCTCModel(
        vocab_size=len(stoi),
        hidden_dim=cfg["model"].get("hidden_dim", 512),
        freeze_backbone=freeze_backbone,
        use_transfer=cfg["model"].get("use_transfer", True),
    ).to(device)
    print(json.dumps({"transfer_loaded": bool(model.transfer_loaded)}))
    print(
        json.dumps(
            {
                "device": str(device),
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": train_batch_size * grad_accum_steps,
                "freeze_backbone": freeze_backbone,
                "unfreeze_epoch": unfreeze_epoch,
                "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            }
        )
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_per = 1e9
    out_dir = Path(cfg["train"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    grad_clip = float(cfg["train"].get("grad_clip", 5.0))
    weight_decay = cfg["train"].get("weight_decay", 1e-4)

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        if freeze_backbone and unfreeze_epoch is not None and epoch == unfreeze_epoch:
            model.unfreeze_backbone()
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["train"]["lr"],
                weight_decay=weight_decay,
            )

        model.train()
        running_loss = 0.0
        oom_skips = 0

        prog = tqdm(train_loader, desc=f"epoch {epoch}", ncols=100)
        opt.zero_grad(set_to_none=True)
        num_train_batches = len(train_loader)
        for step_idx, batch in enumerate(prog, start=1):
            try:
                feats = batch["features"].to(device, non_blocking=True)
                feat_lens = batch["feature_lens"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                target_lens = batch["target_lens"].to(device, non_blocking=True)
                label_weights = batch["label_weights"].to(device, non_blocking=True)

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

                scaler.scale(loss / grad_accum_steps).backward()

                if step_idx % grad_accum_steps == 0 or step_idx == num_train_batches:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                running_loss += float(loss.item())
                prog.set_postfix(loss=f"{loss.item():.4f}", oom=oom_skips)
            except torch.OutOfMemoryError:
                oom_skips += 1
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                prog.set_postfix(loss="oom-skip", oom=oom_skips)
                continue

        val_metrics = evaluate(model, val_loader, itos, blank_id, device)
        avg_loss = running_loss / max(1, len(train_loader))
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_per": val_metrics["per"],
                    "val_cer": val_metrics["cer"],
                    "oom_skips": oom_skips,
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
