"""
Reverse-engineering a BPE tokenizer's training-data mixture.

Idea (Hayase et al., 2024, "Data Mixture Inference"): BPE merges are chosen greedily,
so the merge list is an ordered record of which byte pairs were most frequent in the
training data. At merge step t, the chosen pair must have had the highest count under
the *mixture*:

    sum_i  alpha_i * count_i^(t)(chosen)  >=  sum_i alpha_i * count_i^(t)(other)   for all pairs

where count_i^(t) are pair counts in category i after applying merges 1..t-1. These
are linear constraints on the unknown mixture weights alpha (on the simplex). We
collect them from reference samples of each candidate category, then find the alpha
that violates the fewest/least of them.

Dependencies: Python stdlib + PyTorch (only for the final optimisation step).
Run:  python tokenizer_mixture_inference.py
"""
import random
from collections import Counter

import torch

# ----------------------------------------------------------------------------
# 1. Synthetic "languages" (swap in real text files here if you want)
# ----------------------------------------------------------------------------
LANGS = {
    #            onsets          vowels   codas
    "lang_A": ("ptkmn",       "aiu",   ["", "n", "m"]),
    "lang_B": ("bdgsz",       "eo",    ["", "s", "r"]),
    "lang_C": ("lrfvhtk",     "aeiou", ["", "l", "n", "t"]),
}


def make_vocab(seed, onsets, vowels, codas, n_words=1500):
    rng = random.Random(seed)
    words = set()
    while len(words) < n_words:
        k = rng.choice([1, 2, 2, 3, 3, 4])
        words.add("".join(rng.choice(onsets) + rng.choice(vowels) + rng.choice(codas)
                          for _ in range(k)))
    words = sorted(words)
    rng.shuffle(words)                       # random Zipf rank per language
    return words


def sample_words(vocab, n, seed):
    rng = random.Random(seed)
    zipf = [1 / (r + 1) for r in range(len(vocab))]
    return rng.choices(vocab, zipf, k=n)


def to_word_counts(words):
    """GPT-style pre-tokenisation: ' word' -> tuple of bytes, with counts."""
    return Counter(tuple((" " + w).encode()) for w in words)


# ----------------------------------------------------------------------------
# 2. Minimal BPE trainer (the "black box" whose merge list we get to see)
# ----------------------------------------------------------------------------
def pair_counts(words):
    c = Counter()
    for w, n in words.items():
        for a, b in zip(w, w[1:]):
            c[(a, b)] += n
    return c


def apply_merge(words, pair, new_id):
    out = Counter()
    for w, n in words.items():
        res, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                res.append(new_id); i += 2
            else:
                res.append(w[i]); i += 1
        out[tuple(res)] += n
    return out


def train_bpe(words, n_merges):
    merges = []
    for t in range(n_merges):
        c = pair_counts(words)
        if not c:
            break
        pair = max(c, key=lambda p: (c[p], p))
        merges.append(pair)
        words = apply_merge(words, pair, 256 + t)
    return merges


# ----------------------------------------------------------------------------
# 3. The attack: turn the merge list into linear constraints on the mixture
# ----------------------------------------------------------------------------
def build_constraints(merges, category_words, top_k=40):
    """
    Returns D with shape (num_constraints, num_categories) such that the true
    mixture alpha should satisfy D @ alpha >= 0 (up to sampling noise).
    Each row = normalised_counts(chosen merge) - normalised_counts(competing pair).
    """
    words = [dict(cw) for cw in category_words]
    sizes = [sum(len(w) * n for w, n in cw.items()) for cw in category_words]
    rows = []
    for t, chosen in enumerate(merges):
        # per-category pair counts, normalised by category size (== "per byte" rates)
        counts = [{p: v / s for p, v in pair_counts(w).items()}
                  for w, s in zip(words, sizes)]
        total = lambda p: sum(c.get(p, 0.0) for c in counts)
        vec = lambda p: [c.get(p, 0.0) for c in counts]

        scale = total(chosen)
        if scale > 0:                        # chosen merge must appear in our samples
            v_chosen = vec(chosen)
            rivals = Counter()
            for c in counts:
                for p in c:
                    if p != chosen:
                        rivals[p] = total(p)
            for p, _ in rivals.most_common(top_k):
                rows.append([(a - b) / scale for a, b in zip(v_chosen, vec(p))])

        words = [apply_merge(w, chosen, 256 + t) for w in words]
    return torch.tensor(rows, dtype=torch.float32)


def infer_mixture(D, steps=3000, lr=0.05, margin=1e-3):
    """Minimise total constraint violation over the probability simplex."""
    logits = torch.zeros(D.shape[1], requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        alpha = torch.softmax(logits, dim=0)
        loss = torch.relu(margin - D @ alpha).mean()    # hinge on each constraint
        opt.zero_grad(); loss.backward(); opt.step()
    return torch.softmax(logits, 0).detach()


# ----------------------------------------------------------------------------
# 4. Demo
# ----------------------------------------------------------------------------
def main(true_mix=(0.60, 0.30, 0.10), n_train_words=60_000, n_ref_words=20_000,
         n_merges=300, seed=0):
    names = list(LANGS)
    vocabs = {n: make_vocab(seed + i, *LANGS[n]) for i, n in enumerate(names)}

    # --- the "victim": trained on a secret mixture ---------------------------
    train_words = []
    for i, (n, a) in enumerate(zip(names, true_mix)):
        train_words += sample_words(vocabs[n], int(a * n_train_words), seed=100 + i)
    merges = train_bpe(to_word_counts(train_words), n_merges)
    print(f"victim tokenizer: {len(merges)} merges (all we get to observe)")

    # --- the attacker: only the merge list + fresh reference samples ---------
    ref = [to_word_counts(sample_words(vocabs[n], n_ref_words, seed=200 + i))
           for i, n in enumerate(names)]
    D = build_constraints(merges, ref)
    print(f"built {D.shape[0]} linear constraints over {D.shape[1]} categories")

    est = infer_mixture(D)
    true = torch.tensor(true_mix)
    uniform = torch.full_like(true, 1 / len(names))
    viol = lambda a: (D @ a < 0).float().mean().item()

    print(f"\n{'category':<10}{'true':>8}{'inferred':>10}")
    for n, t, e in zip(names, true, est):
        print(f"{n:<10}{t:>8.3f}{e:>10.3f}")
    print(f"\nL1 error:            {(est - true).abs().sum():.3f}")
    print(f"violated (true mix): {viol(true):.1%}")
    print(f"violated (inferred): {viol(est):.1%}")
    print(f"violated (uniform):  {viol(uniform):.1%}")


if __name__ == "__main__":
    main()
