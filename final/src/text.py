from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.constants import CTC_BLANK, NORMALIZATION_REPLACEMENTS, VALID_IPA_CHARS


@dataclass
class NormalizationResult:
    normalized: str
    unknown_chars: list[str]
    replaced_chars: list[tuple[str, str]]


def normalize_phonetic_text(text: str) -> NormalizationResult:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    out_chars: list[str] = []
    unknown: list[str] = []
    replaced: list[tuple[str, str]] = []

    for ch in text:
        if ch in VALID_IPA_CHARS:
            out_chars.append(ch)
            continue

        mapped = NORMALIZATION_REPLACEMENTS.get(ch, "")
        if mapped:
            replaced.append((ch, mapped))
            for mch in mapped:
                if mch in VALID_IPA_CHARS:
                    out_chars.append(mch)
                else:
                    unknown.append(mch)
        elif ch.isspace():
            out_chars.append(" ")
        else:
            unknown.append(ch)

    normalized = " ".join("".join(out_chars).split())
    return NormalizationResult(normalized=normalized, unknown_chars=unknown, replaced_chars=replaced)


def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    symbols = [CTC_BLANK] + sorted(ch for ch in VALID_IPA_CHARS)
    stoi = {s: i for i, s in enumerate(symbols)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text if ch in stoi]


def decode_ids(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids if i in itos and itos[i] != CTC_BLANK)


def write_vocab(path: Path, stoi: dict[str, int], itos: dict[int, str]) -> None:
    payload = {"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_char_stats(texts: list[str]) -> dict[str, int]:
    c = Counter()
    for t in texts:
        c.update(t)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
