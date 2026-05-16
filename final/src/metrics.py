from __future__ import annotations

from collections import Counter


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def cer(ref_text: str, hyp_text: str) -> float:
    ref = list(ref_text)
    hyp = list(hyp_text)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def per(ref_text: str, hyp_text: str) -> float:
    # PER here treats each non-space IPA character as one phone symbol.
    ref = [c for c in ref_text if c != " "]
    hyp = [c for c in hyp_text if c != " "]
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def phone_tokens(text: str) -> list[str]:
    return [c for c in text if c != " "]


def align_sequences(ref: list[str], hyp: list[str]) -> list[tuple[str, str]]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    aligned: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((ref[i - 1], "<del>"))
            i -= 1
        else:
            aligned.append(("<ins>", hyp[j - 1]))
            j -= 1

    return aligned[::-1]


class PhonemeErrorAnalyzer:
    def __init__(self) -> None:
        self.ref_counts: Counter[str] = Counter()
        self.hyp_counts: Counter[str] = Counter()
        self.correct_counts: Counter[str] = Counter()
        self.substitutions: Counter[tuple[str, str]] = Counter()
        self.deletions: Counter[str] = Counter()
        self.insertions: Counter[str] = Counter()
        self.total_ref_tokens = 0
        self.total_hyp_tokens = 0
        self.total_edits = 0
        self.num_samples = 0

    def update(self, ref_text: str, hyp_text: str) -> None:
        ref = phone_tokens(ref_text)
        hyp = phone_tokens(hyp_text)
        self.ref_counts.update(ref)
        self.hyp_counts.update(hyp)
        self.total_ref_tokens += len(ref)
        self.total_hyp_tokens += len(hyp)
        self.num_samples += 1

        for ref_token, hyp_token in align_sequences(ref, hyp):
            if ref_token == hyp_token:
                self.correct_counts[ref_token] += 1
            elif hyp_token == "<del>":
                self.deletions[ref_token] += 1
                self.total_edits += 1
            elif ref_token == "<ins>":
                self.insertions[hyp_token] += 1
                self.total_edits += 1
            else:
                self.substitutions[(ref_token, hyp_token)] += 1
                self.total_edits += 1

    @staticmethod
    def _counter_rows(counter: Counter, limit: int | None = None) -> list[dict[str, object]]:
        rows = []
        items = counter.most_common(limit)
        for key, count in items:
            if isinstance(key, tuple):
                rows.append({"ref": key[0], "hyp": key[1], "count": int(count)})
            else:
                rows.append({"phoneme": key, "count": int(count)})
        return rows

    def per_phoneme_rows(self) -> list[dict[str, object]]:
        rows = []
        for phoneme, ref_count in self.ref_counts.items():
            correct = self.correct_counts[phoneme]
            substitutions = sum(count for (ref, _), count in self.substitutions.items() if ref == phoneme)
            deletions = self.deletions[phoneme]
            errors = substitutions + deletions
            rows.append(
                {
                    "phoneme": phoneme,
                    "ref_count": int(ref_count),
                    "hyp_count": int(self.hyp_counts[phoneme]),
                    "correct": int(correct),
                    "substitutions": int(substitutions),
                    "deletions": int(deletions),
                    "insertions": int(self.insertions[phoneme]),
                    "recall": float(correct / ref_count) if ref_count else 0.0,
                    "error_rate": float(errors / ref_count) if ref_count else 0.0,
                }
            )
        return sorted(rows, key=lambda row: (row["recall"], -row["ref_count"], row["phoneme"]))

    def summary(self, top_k: int = 25, min_ref_count: int = 5) -> dict[str, object]:
        per_phoneme = self.per_phoneme_rows()
        eligible = [row for row in per_phoneme if int(row["ref_count"]) >= min_ref_count]
        return {
            "num_samples": int(self.num_samples),
            "total_ref_tokens": int(self.total_ref_tokens),
            "total_hyp_tokens": int(self.total_hyp_tokens),
            "total_edits": int(self.total_edits),
            "micro_per_from_alignment": float(self.total_edits / max(1, self.total_ref_tokens)),
            "min_ref_count_for_worst_lists": int(min_ref_count),
            "worst_recall": eligible[:top_k],
            "most_common_substitutions": self._counter_rows(self.substitutions, top_k),
            "most_common_deletions": self._counter_rows(self.deletions, top_k),
            "most_common_insertions": self._counter_rows(self.insertions, top_k),
            "per_phoneme": per_phoneme,
        }
