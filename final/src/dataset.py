from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.augment import SpecAugment, maybe_wave_augment
from src.features import LogMelFrontend, load_audio
from src.text import encode_text


class PhonemeDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        stoi: dict[str, int],
        sample_rate: int = 16000,
        train: bool = False,
        use_specaugment: bool = True,
    ) -> None:
        self.rows = self._read_jsonl(manifest_path)
        self.stoi = stoi
        self.train = train
        self.frontend = LogMelFrontend(sample_rate=sample_rate)
        self.specaug = SpecAugment() if train and use_specaugment else None

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        out: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        waveform, _ = load_audio(row["audio_path"])
        if self.train:
            waveform = maybe_wave_augment(waveform)

        feat = self.frontend(waveform)
        if self.specaug is not None:
            feat = self.specaug(feat)

        text = row.get("phonetic_text", "")
        target = torch.tensor(encode_text(text, self.stoi), dtype=torch.long)
        return {
            "utt_id": row["utterance_id"],
            "feature": feat,
            "feature_len": torch.tensor(feat.size(0), dtype=torch.long),
            "target": target,
            "target_len": torch.tensor(target.numel(), dtype=torch.long),
            "label_weight": torch.tensor(float(row.get("label_weight", 1.0)), dtype=torch.float32),
            "text": text,
        }


def ctc_collate(batch: list[dict]) -> dict:
    feats = [b["feature"] for b in batch]
    feat_lens = torch.stack([b["feature_len"] for b in batch])
    targets = [b["target"] for b in batch]
    target_lens = torch.stack([b["target_len"] for b in batch])
    label_weights = torch.stack([b["label_weight"] for b in batch])

    padded_feats = pad_sequence(feats, batch_first=True)
    flat_targets = torch.cat(targets, dim=0) if targets else torch.empty(0, dtype=torch.long)

    return {
        "utt_ids": [b["utt_id"] for b in batch],
        "features": padded_feats,
        "feature_lens": feat_lens,
        "targets": flat_targets,
        "target_lens": target_lens,
        "label_weights": label_weights,
        "texts": [b["text"] for b in batch],
    }
