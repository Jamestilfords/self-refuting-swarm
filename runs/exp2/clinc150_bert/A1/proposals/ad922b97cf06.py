import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy and pooled-space Mahalanobis novelty are converted to train-calibrated upper-tail "
    "percentiles before a geometric fusion, so queries that are jointly unusual in classifier "
    "evidence and representation compatibility receive the largest OOS scores."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0081"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def upper_tail_reference(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        n = ref.size
        positions = np.searchsorted(ref, values, side="right")
        return (positions + 0.5) / (n + 1.0)

    energy_p = upper_tail_reference(train_energy, eval_energy)
    dist_p = upper_tail_reference(train_dist, eval_dist)

    score_values = np.sqrt(np.maximum(energy_p, 1e-12) * np.maximum(dist_p, 1e-12))
    return np.asarray(score_values, dtype=float)
