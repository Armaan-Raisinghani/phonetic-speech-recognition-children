"""
Phone-level n-gram language model for IPA sequences.

Trains on character-level IPA transcriptions from the training manifest.
Uses Witten-Bell smoothing so that unseen n-grams get nonzero probability.
The LM is used at CTC beam-search decoding time, not during training.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


class PhoneNgramLM:
    """Character-level n-gram LM over IPA phone sequences."""

    def __init__(self, order: int = 3) -> None:
        if order < 1:
            raise ValueError("n-gram order must be >= 1")
        self.order = int(order)
        # counts[context_tuple] -> Counter[next_char]
        self.counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()
        self._bos = "<s>"
        self._eos = "</s>"

    def train(self, texts: list[str]) -> None:
        """Train counts from a list of IPA transcription strings."""
        for text in texts:
            chars = list(text.replace(" ", ""))
            if not chars:
                continue
            # pad with bos/eos markers
            padded = [self._bos] * (self.order - 1) + chars + [self._eos]
            self.vocab.update(chars)
            for i in range(self.order - 1, len(padded)):
                context = tuple(padded[i - self.order + 1 : i])
                token = padded[i]
                self.counts[context][token] += 1

    def log_prob(self, char: str, context: tuple[str, ...]) -> float:
        """Log probability of *char* given *context* (Witten-Bell smoothed).

        Falls back to shorter contexts when needed (backoff).
        """
        for n in range(min(self.order - 1, len(context)), -1, -1):
            ctx = context[-n:] if n > 0 else ()
            counter = self.counts.get(ctx)
            if counter is None:
                continue
            total = sum(counter.values())
            count = counter.get(char, 0)
            if total == 0:
                continue
            # witten-bell: lambda = 1 - T / (T + N), where T = num distinct types, N = total count
            num_types = len(counter)
            lam = 1.0 - num_types / (num_types + total)
            if count > 0:
                return math.log(lam * count / total + 1e-10)
            else:
                # backoff with weight (1 - lambda), distribute uniformly over unseen
                unseen = max(1, len(self.vocab) - num_types)
                return math.log((1 - lam) / unseen + 1e-10)
        # absolute fallback: uniform
        return math.log(1.0 / max(1, len(self.vocab)))

    def score_sequence(self, chars: list[str]) -> float:
        """Total log probability of a phone sequence."""
        padded = [self._bos] * (self.order - 1) + chars + [self._eos]
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            context = tuple(padded[i - self.order + 1 : i])
            total += self.log_prob(padded[i], context)
        return total

    def save(self, path: Path) -> None:
        """Save n-gram counts to a JSON file."""
        payload = {
            "order": self.order,
            "vocab": sorted(self.vocab),
            "counts": {
                "|".join(ctx): dict(counter)
                for ctx, counter in self.counts.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PhoneNgramLM":
        """Load n-gram counts from a JSON file."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        lm = cls(order=int(payload["order"]))
        lm.vocab = set(payload["vocab"])
        for ctx_str, counter_dict in payload["counts"].items():
            ctx = tuple(ctx_str.split("|")) if ctx_str else ()
            lm.counts[ctx] = Counter(counter_dict)
        return lm


def train_phone_lm_from_manifest(
    manifest_path: Path,
    order: int = 3,
) -> PhoneNgramLM:
    """Train a phone n-gram LM from a JSONL manifest's phonetic_text field."""
    texts: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("phonetic_text", "")
            if text:
                texts.append(text)
    lm = PhoneNgramLM(order=order)
    lm.train(texts)
    return lm
