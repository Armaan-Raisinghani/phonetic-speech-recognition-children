from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import soundfile as sf


@dataclass
class ManifestRow:
    utterance_id: str
    child_id: str
    session_id: str
    audio_path: str
    audio_duration_sec: float
    age_bucket: str
    phonetic_text: str
    split: str
    is_pseudo: bool
    label_weight: float


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_audio_files(audio_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for p in audio_dir.rglob("*.flac"):
        mapping[p.stem] = str(p)
    return mapping


def audit_labeled_rows(rows: list[dict], root_dir: Path) -> tuple[list[dict], list[dict]]:
    valid: list[dict] = []
    invalid: list[dict] = []

    for row in rows:
        path = root_dir / row["audio_path"]
        candidate = dict(row)
        candidate["resolved_audio_path"] = str(path)
        if not path.exists():
            candidate["error"] = "missing_audio"
            invalid.append(candidate)
            continue
        try:
            info = sf.info(str(path))
            candidate["num_frames"] = int(info.frames)
            candidate["sample_rate"] = int(info.samplerate)
        except Exception as exc:  # pragma: no cover
            candidate["error"] = f"decode_error:{exc}"
            invalid.append(candidate)
            continue

        valid.append(candidate)

    return valid, invalid


def build_unlabeled_rows(audio_index: dict[str, str], labeled_ids: set[str]) -> list[dict]:
    out: list[dict] = []
    for utt_id, path in audio_index.items():
        if utt_id in labeled_ids:
            continue
        out.append(
            {
                "utterance_id": utt_id,
                "audio_path": path,
                "split": "unlabeled",
                "is_pseudo": False,
                "label_weight": 0.0,
            }
        )
    return out


def speaker_disjoint_split(
    rows: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1337,
) -> dict[str, str]:
    rng = random.Random(seed)
    by_child: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_child[row["child_id"]].append(row)

    child_ids = list(by_child.keys())
    rng.shuffle(child_ids)

    n = len(child_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_children = set(child_ids[:n_train])
    val_children = set(child_ids[n_train : n_train + n_val])

    split_by_utt: dict[str, str] = {}
    for row in rows:
        cid = row["child_id"]
        if cid in train_children:
            split = "train"
        elif cid in val_children:
            split = "val"
        else:
            split = "test"
        split_by_utt[row["utterance_id"]] = split
    return split_by_utt


def split_stats(rows: list[dict]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        split = r["split"]
        out.setdefault(split, {"samples": 0})
        out[split]["samples"] += 1
    return out
