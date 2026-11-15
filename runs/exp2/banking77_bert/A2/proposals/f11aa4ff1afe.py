import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and full encoder-space Mahalanobis distance provide complementary logit and representation novelty signals. Smoothed train empirical-CDF log-odds put their upper tails on a common scale before additive fusion."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0083", "m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def tail_log_odds(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        n = ref.size
        ranks = np.searchsorted(ref, np.asarray(values, dtype=float), side="right")
        p = (ranks + 0.5) / (n + 1.0)
        return np.log(p) - np.log1p(-p)

    energy_tail = tail_log_odds(train_energy, eval_energy)
    distance_tail = tail_log_odds(train_dist, eval_dist)
    return energy_tail + distance_tail
