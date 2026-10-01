"""
bpe_provenance.py

Heuristic estimators for reasoning about how a BPE tokenizer was trained.
Standard library only (math); no torch required.

Conventions
-----------
* Tokens are strings in the tokenizer's own symbol alphabet (e.g. GPT-2 style
  'Ġthe' or SentencePiece '▁the'); each character is treated as one base symbol.
* `corpus_size` is the total number of base symbols the BPE trainer saw.
* Merge k (1-indexed) is modelled as having been chosen when its pair count was
      c_k = first_merge_frac * corpus_size * k ** (-zipf_alpha)
  BPE always merges the most frequent pair, so counts are non-increasing in k.
  A power law is a reasonable first-order fit; calibrate with `fit_zipf` if you
  have real merge counts from a training log.
* Tokens are assumed to be complete pretokens (whole words / word pieces that
  the pretokenizer keeps intact). Tokens that only occur *inside* longer words
  face extra competition from merges that straddle their boundaries, so treat
  the frequency estimates for those as lower bounds.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class BPETrainingEstimator:
    def __init__(
        self,
        vocab: Mapping[str, int],
        merges: Optional[Sequence[Tuple[str, str]]] = None,
        corpus_size: float = 1e9,
        zipf_alpha: float = 1.0,
        first_merge_frac: float = 0.03,
    ):
        """
        vocab:   token string -> id.
        merges:  ordered merge list [(left, right), ...]. If omitted, it is
                 reconstructed from the vocab, assuming ids follow merge order
                 (true for GPT-2 / tiktoken-style vocabularies).
        """
        self.vocab: Dict[str, int] = dict(vocab)
        self.corpus_size = float(corpus_size)
        self.zipf_alpha = float(zipf_alpha)
        self.first_merge_frac = float(first_merge_frac)

        self.merges: List[Tuple[str, str]] = (
            [tuple(m) for m in merges] if merges is not None else self._infer_merges()
        )
        self.num_merges = len(self.merges)
        self._pair_rank: Dict[Tuple[str, str], int] = {}
        self._result_rank: Dict[str, int] = {}  # merged string -> earliest rank producing it
        for rank, (a, b) in enumerate(self.merges, start=1):
            self._pair_rank.setdefault((a, b), rank)
            self._result_rank.setdefault(a + b, rank)

    # ------------------------------------------------------------------ #
    # Tokenizer internals
    # ------------------------------------------------------------------ #
    def _infer_merges(self) -> List[Tuple[str, str]]:
        """Recover merge order from token ids: split each token into the two
        lower-id vocab entries whose later-appearing half has the smallest id."""
        ids = self.vocab
        merges = []
        for tok, tid in sorted(ids.items(), key=lambda kv: kv[1]):
            if len(tok) < 2:
                continue
            best = None
            for i in range(1, len(tok)):
                a, b = tok[:i], tok[i:]
                if a in ids and b in ids and ids[a] < tid and ids[b] < tid:
                    score = max(ids[a], ids[b])
                    if best is None or score < best[0]:
                        best = (score, a, b)
            if best is not None:
                merges.append((best[1], best[2]))
        return merges

    def _segment(self, token: str) -> List[str]:
        """Apply the learned merges (lowest rank first) to a string."""
        symbols = list(token)
        while len(symbols) > 1:
            best_rank, best_i = None, None
            for i in range(len(symbols) - 1):
                r = self._pair_rank.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]
        return symbols

    # ------------------------------------------------------------------ #
    # Frequency model
    # ------------------------------------------------------------------ #
    def merge_count_at_rank(self, rank: float, corpus_size: Optional[float] = None) -> float:
        """Modelled pair count at which merge number `rank` was chosen."""
        n = self.corpus_size if corpus_size is None else corpus_size
        return self.first_merge_frac * n * max(rank, 1.0) ** (-self.zipf_alpha)

    def fit_zipf(self, merge_counts: Sequence[float]) -> Tuple[float, float]:
        """Least-squares fit of log(count) = log(C1) - alpha*log(rank) to observed
        per-merge pair counts (in merge order). Updates and returns
        (zipf_alpha, first_merge_frac)."""
        pts = [(math.log(k), math.log(c)) for k, c in enumerate(merge_counts, 1) if c > 0]
        n = len(pts)
        if n < 2:
            raise ValueError("Need at least two positive merge counts to fit.")
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
        intercept = my - slope * mx
        self.zipf_alpha = -slope
        self.first_merge_frac = math.exp(intercept) / self.corpus_size
        return self.zipf_alpha, self.first_merge_frac

    # ------------------------------------------------------------------ #
    # 1) How likely was this tokenizer trained on a dataset containing these?
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_null_atomic_prob(length: int) -> float:
        """Prior guess: chance an arbitrary domain string of `length` chars is a
        single token in a generic tokenizer (0.7 at 2 chars, decaying 0.6x/char)."""
        return min(0.95, 0.7 * 0.6 ** (length - 2))

    def estimate_training_likelihood(
        self,
        tokens: Sequence[str],
        prior: float = 0.5,
        p_atomic_if_custom: float = 0.85,
        null_atomic_prob: Optional[Callable[[int], float]] = None,
        reference_freqs: Optional[Mapping[str, float]] = None,
        llr_cap: float = 3.0,
    ) -> Dict:
        """
        Posterior probability that the tokenizer was trained on a dataset that
        contains `tokens` at meaningful frequency (vs. a generic corpus).

        Evidence per token (log-likelihood ratio, capped at +/-llr_cap):
          * Whole token in vocab: the longer it is, the less likely it is to
            be a single token by chance under the generic null
            (`null_atomic_prob(length)`), plus a bonus for early merge ranks
            (early = frequent).
          * If `reference_freqs` (token -> expected occurrences per symbol in a
            generic corpus) has the token: use log(implied_count / expected_count)
            instead. Much stronger, and needs no length prior.
          * Token fragments into several pieces: evidence against.
        Per-token evidence is combined as if independent, so keep `llr_cap`
        modest; related tokens (e.g. one domain's vocabulary) are correlated.
        """
        if not 0.0 < prior < 1.0:
            raise ValueError("prior must be in (0, 1)")
        null_fn = null_atomic_prob or self._default_null_atomic_prob

        per_token = []
        total_llr = 0.0
        for tok in dict.fromkeys(tokens):  # dedupe, keep order
            L = len(tok)
            if L < 2:
                continue  # single characters carry no information
            pieces = self._segment(tok)
            atomic = len(pieces) == 1
            p0 = min(max(null_fn(L), 1e-6), 0.999)
            rank = self._result_rank.get(tok) if atomic else None
            implied = self.merge_count_at_rank(rank) if rank else None

            ref = reference_freqs.get(tok) if reference_freqs else None
            if atomic and implied and ref and ref > 0:
                llr = math.log(implied / (ref * self.corpus_size))
                basis = "implied_vs_reference_count"
            elif atomic:
                u = min(rank / max(self.num_merges, 1), 0.99)
                llr = math.log(p_atomic_if_custom / p0) + math.log(2.0 * (1.0 - u))
                basis = "length_null+merge_rank"
            else:
                llr = math.log((1.0 - p_atomic_if_custom) / (1.0 - p0))
                basis = "fragmented"
            llr = max(-llr_cap, min(llr_cap, llr))
            total_llr += llr
            per_token.append(
                {
                    "token": tok,
                    "atomic": atomic,
                    "pieces": pieces,
                    "merge_rank": rank,
                    "implied_train_count": implied,
                    "llr": llr,
                    "basis": basis,
                }
            )

        log_odds = math.log(prior / (1.0 - prior)) + total_llr
        n = len(per_token)
        return {
            "probability": _sigmoid(log_odds),
            "log_odds": log_odds,
            "n_tokens_scored": n,
            "atomic_fraction": (sum(t["atomic"] for t in per_token) / n) if n else None,
            "per_token": per_token,
        }

    # ------------------------------------------------------------------ #
    # 2) How often must tokens occur for BPE to keep their characters together?
    # ------------------------------------------------------------------ #
    def estimate_required_frequency(
        self,
        tokens: Sequence[str],
        num_merges: Optional[int] = None,
        corpus_size: Optional[float] = None,
        safety_margin: float = 1.0,
        min_frequency_floor: int = 2,
        background_counts: Optional[Mapping[str, float]] = None,
    ) -> Dict:
        """
        Minimum training-corpus occurrences for each token to end up as a single
        BPE token with a merge budget of `num_merges` (default: this tokenizer's).

        Reasoning: while a word is still split, every adjacent pair inside it has
        count >= its own frequency f, so BPE keeps merging it until the global
        best pair count drops below f. The word therefore completes within the
        budget iff f >= c(M_eff), the pair count of the last affordable merge.
        The tokens themselves consume L-1 merge slots each (upper bound; shared
        sub-pieces make this smaller), which shrinks M_eff and raises the bar.

        background_counts: token -> occurrences already supplied by other
            contexts that feed the same final merge; subtracts from the need.
        safety_margin: multiplier for tie-breaking/sampling noise (>1 = safer).
        """
        M = int(num_merges if num_merges is not None else self.num_merges)
        N = float(corpus_size if corpus_size is not None else self.corpus_size)
        uniq = [t for t in dict.fromkeys(tokens) if len(t) >= 2]

        slots = sum(len(t) - 1 for t in uniq)
        m_eff = max(1, M - slots)
        threshold = self.merge_count_at_rank(m_eff, N) * safety_margin

        per_token = {}
        for tok in uniq:
            bg = float(background_counts.get(tok, 0.0)) if background_counts else 0.0
            need = max(min_frequency_floor, math.ceil(max(threshold - bg, 0.0)))
            rank = self._result_rank.get(tok)
            current = self.merge_count_at_rank(rank, N) if rank else None
            per_token[tok] = {
                "min_count": need,
                "per_million_symbols": need / N * 1e6,
                "merges_needed": len(tok) - 1,
                "currently_atomic": rank is not None,
                "implied_current_count": current,
            }

        return {
            "threshold_pair_count": threshold,
            "effective_merge_budget": m_eff,
            "merge_slots_used_by_tokens": slots,
            "budget_exhausted": slots >= M,
            "per_token": per_token,
        }


# ---------------------------------------------------------------------- #
# Demo: train a toy BPE on a Zipfian synthetic corpus, then probe it.
# ---------------------------------------------------------------------- #
def _toy_bpe_train(word_freqs: Mapping[str, int], num_merges: int):
    from collections import Counter

    words = {tuple(w): f for w, f in word_freqs.items()}
    merges, counts = [], []
    for _ in range(num_merges):
        pc: Counter = Counter()
        for w, f in words.items():
            for a, b in zip(w, w[1:]):
                pc[(a, b)] += f
        if not pc:
            break
        (a, b), c = pc.most_common(1)[0]
        merges.append((a, b))
        counts.append(c)
        new: Dict[tuple, int] = {}
        for w, f in words.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + f
        words = new
    chars = sorted({c for w in word_freqs for c in w})
    vocab = {c: i for i, c in enumerate(chars)}
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    return vocab, merges, counts, sum(f * len(w) for w, f in word_freqs.items())


if __name__ == "__main__":
    import random

    rng = random.Random(0)
    alphabet = "abcdefghijklmnopqrst"
    corpus = {}
    for r in range(1, 3001):
        w = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 8)))
        corpus[w] = corpus.get(w, 0) + max(1, int(20000 / r))
    corpus["quantum"] = 600  # a "domain" word, upweighted

    vocab, merges, counts, n_symbols = _toy_bpe_train(corpus, num_merges=400)
    est = BPETrainingEstimator(vocab, merges=merges, corpus_size=n_symbols)
    print("fitted (alpha, C1/N):", est.fit_zipf(counts))

    res = est.estimate_training_likelihood(["quantum", "zzzzzz", "abc"])
    print("P(custom dataset) =", round(res["probability"], 3))
    for t in res["per_token"]:
        print("  ", t["token"], t["pieces"], round(t["llr"], 2), t["basis"])

    req = est.estimate_required_frequency(["quantum", "hamiltonian"])
    print("pair-count threshold:", round(req["threshold_pair_count"], 1))
    for tok, d in req["per_token"].items():
        print("  ", tok, d["min_count"], "occurrences")
