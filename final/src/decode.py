from __future__ import annotations

import math

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


def _logsumexp(*values: float) -> float:
    finite_vals = [v for v in values if v != -math.inf]
    if not finite_vals:
        return -math.inf
    max_val = max(finite_vals)
    return max_val + math.log(sum(math.exp(v - max_val) for v in finite_vals))


def ctc_prefix_beam_search_decode(
    logits: torch.Tensor,
    blank_id: int = 0,
    beam_size: int = 8,
) -> list[list[int]]:
    # logits: [B, T, C]
    log_probs = torch.log_softmax(logits.detach().cpu(), dim=-1)
    decoded: list[list[int]] = []

    for seq_log_probs in log_probs:
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}

        for frame in seq_log_probs:
            next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
            frame_vals = frame.tolist()

            for prefix, (prob_blank, prob_non_blank) in beams.items():
                for token, token_log_prob in enumerate(frame_vals):
                    next_prob_blank, next_prob_non_blank = next_beams.get(prefix, (-math.inf, -math.inf))

                    if token == blank_id:
                        next_beams[prefix] = (
                            _logsumexp(next_prob_blank, prob_blank + token_log_prob, prob_non_blank + token_log_prob),
                            next_prob_non_blank,
                        )
                        continue

                    last_token = prefix[-1] if prefix else None
                    extended_prefix = prefix + (token,)

                    if token == last_token:
                        repeated_prob_blank, repeated_prob_non_blank = next_beams.get(prefix, (-math.inf, -math.inf))
                        next_beams[prefix] = (
                            repeated_prob_blank,
                            _logsumexp(repeated_prob_non_blank, prob_non_blank + token_log_prob),
                        )

                        ext_prob_blank, ext_prob_non_blank = next_beams.get(extended_prefix, (-math.inf, -math.inf))
                        next_beams[extended_prefix] = (
                            ext_prob_blank,
                            _logsumexp(ext_prob_non_blank, prob_blank + token_log_prob),
                        )
                    else:
                        ext_prob_blank, ext_prob_non_blank = next_beams.get(extended_prefix, (-math.inf, -math.inf))
                        next_beams[extended_prefix] = (
                            ext_prob_blank,
                            _logsumexp(ext_prob_non_blank, prob_blank + token_log_prob, prob_non_blank + token_log_prob),
                        )

            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda item: _logsumexp(item[1][0], item[1][1]),
                    reverse=True,
                )[:beam_size]
            )

        best_prefix, _ = max(beams.items(), key=lambda item: _logsumexp(item[1][0], item[1][1]))
        decoded.append(list(best_prefix))

    return decoded


def decode_logits(
    logits: torch.Tensor,
    blank_id: int = 0,
    strategy: str = "beam",
    beam_size: int = 8,
) -> list[list[int]]:
    normalized = strategy.strip().lower()
    if normalized == "greedy":
        return greedy_ctc_decode(logits, blank_id=blank_id)
    if normalized == "beam":
        return ctc_prefix_beam_search_decode(logits, blank_id=blank_id, beam_size=beam_size)
    raise ValueError(f"Unsupported decode strategy: {strategy}")


def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Mean max posterior over frames, shape [B]
    probs = torch.softmax(logits, dim=-1)
    frame_conf, _ = probs.max(dim=-1)
    return frame_conf.mean(dim=-1)
