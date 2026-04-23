from __future__ import annotations

import torch


def greedy_ctc_decode(logits: torch.Tensor, blank_id: int = 0) -> list[list[int]]:
    # logits: [B, T, C]
    pred = logits.argmax(dim=-1)
    decoded: list[list[int]] = []
    for seq in pred:
        out: list[int] = []
        prev = None
        for token in seq.tolist():
            if token == blank_id:
                prev = token
                continue
            if token == prev:
                continue
            out.append(token)
            prev = token
        decoded.append(out)
    return decoded


def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Mean max posterior over frames, shape [B]
    probs = torch.softmax(logits, dim=-1)
    frame_conf, _ = probs.max(dim=-1)
    return frame_conf.mean(dim=-1)
