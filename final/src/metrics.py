from __future__ import annotations


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
