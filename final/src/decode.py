from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.phone_lm import PhoneNgramLM


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


def ctc_prefix_beam_search_lm_decode(
    logits: torch.Tensor,
    blank_id: int = 0,
    beam_size: int = 16,
    lm: "PhoneNgramLM | None" = None,
    itos: dict[int, str] | None = None,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
) -> list[list[int]]:
    """CTC prefix beam search with optional phone LM shallow fusion and insertion bonus.

    The total score for each beam is:
        acoustic_score + lm_weight * lm_score + insertion_bonus * num_emitted_tokens

    The insertion bonus encourages the model to emit more phones, directly
    combating CTC under-emission / deletion behavior.
    """
    log_probs = torch.log_softmax(logits.detach().cpu(), dim=-1)
    decoded: list[list[int]] = []

    use_lm = lm is not None and itos is not None and lm_weight > 0.0
    bos_marker = "<s>"

    for seq_log_probs in log_probs:
        # each beam: prefix -> (acoustic_blank, acoustic_non_blank, lm_score, num_tokens)
        beams: dict[tuple[int, ...], tuple[float, float, float, int]] = {
            (): (0.0, -math.inf, 0.0, 0)
        }

        for frame in seq_log_probs:
            next_beams: dict[tuple[int, ...], tuple[float, float, float, int]] = {}
            frame_vals = frame.tolist()

            for prefix, (prob_blank, prob_non_blank, lm_score, n_tokens) in beams.items():
                for token, token_log_prob in enumerate(frame_vals):

                    if token == blank_id:
                        cur = next_beams.get(prefix, (-math.inf, -math.inf, 0.0, 0))
                        next_beams[prefix] = (
                            _logsumexp(cur[0], prob_blank + token_log_prob, prob_non_blank + token_log_prob),
                            cur[1],
                            lm_score,
                            n_tokens,
                        )
                        continue

                    last_token = prefix[-1] if prefix else None
                    extended_prefix = prefix + (token,)

                    if token == last_token:
                        # repeat: extend only via blank path
                        cur = next_beams.get(prefix, (-math.inf, -math.inf, 0.0, 0))
                        next_beams[prefix] = (
                            cur[0],
                            _logsumexp(cur[1], prob_non_blank + token_log_prob),
                            lm_score,
                            n_tokens,
                        )

                        # compute lm score for the extended prefix
                        ext_lm_score = lm_score
                        if use_lm:
                            char = itos.get(token, "")
                            context_chars = [itos.get(t, "") for t in prefix[-(lm.order - 1):]]
                            if not context_chars:
                                context_chars = [bos_marker]
                            ext_lm_score = lm_score + lm.log_prob(char, tuple(context_chars))

                        cur_ext = next_beams.get(extended_prefix, (-math.inf, -math.inf, 0.0, 0))
                        next_beams[extended_prefix] = (
                            cur_ext[0],
                            _logsumexp(cur_ext[1], prob_blank + token_log_prob),
                            ext_lm_score,
                            n_tokens + 1,
                        )
                    else:
                        # new token emission
                        ext_lm_score = lm_score
                        if use_lm:
                            char = itos.get(token, "")
                            context_chars = [itos.get(t, "") for t in prefix[-(lm.order - 1):]]
                            if not context_chars:
                                context_chars = [bos_marker]
                            ext_lm_score = lm_score + lm.log_prob(char, tuple(context_chars))

                        cur_ext = next_beams.get(extended_prefix, (-math.inf, -math.inf, 0.0, 0))
                        next_beams[extended_prefix] = (
                            cur_ext[0],
                            _logsumexp(cur_ext[1], prob_blank + token_log_prob, prob_non_blank + token_log_prob),
                            ext_lm_score,
                            n_tokens + 1,
                        )

            # prune: score = acoustic + lm_weight * lm + insertion_bonus * length
            def _beam_score(item: tuple[tuple[int, ...], tuple[float, float, float, int]]) -> float:
                _, (pb, pnb, lms, ntok) = item
                acoustic = _logsumexp(pb, pnb)
                return acoustic + lm_weight * lms + insertion_bonus * ntok

            beams = dict(
                sorted(next_beams.items(), key=_beam_score, reverse=True)[:beam_size]
            )

        # final selection with full score
        best_prefix, _ = max(
            beams.items(),
            key=lambda item: (
                _logsumexp(item[1][0], item[1][1])
                + lm_weight * item[1][2]
                + insertion_bonus * item[1][3]
            ),
        )
        decoded.append(list(best_prefix))

    return decoded


def decode_logits(
    logits: torch.Tensor,
    blank_id: int = 0,
    strategy: str = "beam",
    beam_size: int = 8,
    lm: "PhoneNgramLM | None" = None,
    itos: dict[int, str] | None = None,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
) -> list[list[int]]:
    normalized = strategy.strip().lower()
    if normalized == "greedy":
        return greedy_ctc_decode(logits, blank_id=blank_id)
    if normalized == "beam":
        # use LM-weighted search if LM or insertion bonus is configured
        if (lm is not None and lm_weight > 0.0) or insertion_bonus != 0.0:
            return ctc_prefix_beam_search_lm_decode(
                logits,
                blank_id=blank_id,
                beam_size=beam_size,
                lm=lm,
                itos=itos,
                lm_weight=lm_weight,
                insertion_bonus=insertion_bonus,
            )
        return ctc_prefix_beam_search_decode(logits, blank_id=blank_id, beam_size=beam_size)
    raise ValueError(f"Unsupported decode strategy: {strategy}")


def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Mean max posterior over frames, shape [B]
    probs = torch.softmax(logits, dim=-1)
    frame_conf, _ = probs.max(dim=-1)
    return frame_conf.mean(dim=-1)
