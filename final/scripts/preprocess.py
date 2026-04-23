#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import (
    audit_labeled_rows,
    build_unlabeled_rows,
    index_audio_files,
    read_jsonl,
    speaker_disjoint_split,
    split_stats,
    write_jsonl,
)
from src.text import build_vocab, collect_char_stats, normalize_phonetic_text, write_vocab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--labels", type=Path, default=Path("train_phon_transcripts.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio"))
    p.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    labels_path = (root / args.labels).resolve()
    audio_dir = (root / args.audio_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(labels_path)
    valid_rows, invalid_rows = audit_labeled_rows(rows, root)

    normalized_rows: list[dict] = []
    dropped_rows: list[dict] = []

    for row in valid_rows:
        norm = normalize_phonetic_text(row["phonetic_text"])
        candidate = dict(row)
        candidate["phonetic_text_raw"] = row["phonetic_text"]
        candidate["phonetic_text"] = norm.normalized
        candidate["unknown_chars"] = norm.unknown_chars
        candidate["replaced_chars"] = norm.replaced_chars
        candidate["audio_path"] = row["resolved_audio_path"]

        if not candidate["phonetic_text"]:
            candidate["error"] = "empty_after_normalization"
            dropped_rows.append(candidate)
            continue
        if norm.unknown_chars:
            candidate["error"] = "unknown_ipa_chars"
            dropped_rows.append(candidate)
            continue
        normalized_rows.append(candidate)

    split_map = speaker_disjoint_split(
        normalized_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    manifests = {"train": [], "val": [], "test": []}
    for row in normalized_rows:
        split = split_map[row["utterance_id"]]
        item = {
            "utterance_id": row["utterance_id"],
            "child_id": row["child_id"],
            "session_id": row["session_id"],
            "audio_path": row["audio_path"],
            "audio_duration_sec": row["audio_duration_sec"],
            "age_bucket": row["age_bucket"],
            "phonetic_text": row["phonetic_text"],
            "split": split,
            "is_pseudo": False,
            "label_weight": 1.0,
        }
        manifests[split].append(item)

    for split, split_rows in manifests.items():
        write_jsonl(out_dir / f"{split}.jsonl", split_rows)

    all_labeled_ids = {r["utterance_id"] for r in normalized_rows}
    audio_index = index_audio_files(audio_dir)
    unlabeled = build_unlabeled_rows(audio_index, all_labeled_ids)
    write_jsonl(out_dir / "unlabeled.jsonl", unlabeled)

    stoi, itos = build_vocab()
    write_vocab(out_dir / "vocab.json", stoi, itos)

    report = {
        "num_input_rows": len(rows),
        "num_valid_audio_rows": len(valid_rows),
        "num_invalid_audio_rows": len(invalid_rows),
        "num_dropped_after_normalization": len(dropped_rows),
        "num_final_labeled": len(normalized_rows),
        "num_unlabeled": len(unlabeled),
        "split_stats": split_stats(
            manifests["train"] + manifests["val"] + manifests["test"]
        ),
        "char_stats": collect_char_stats([r["phonetic_text"] for r in normalized_rows]),
    }

    (out_dir / "invalid_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in invalid_rows),
        encoding="utf-8",
    )
    (out_dir / "dropped_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in dropped_rows),
        encoding="utf-8",
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
