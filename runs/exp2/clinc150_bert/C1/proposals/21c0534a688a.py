import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Training-set empirical percentiles put logit energy and pooled-feature Mahalanobis distance on a shared robust scale. Queries that are jointly atypical in confidence and representation space receive high scores."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def percentile_from_train(train_values, values):
        ref = np.sort(np.nan_to_num(train_values, nan=0.0, posinf=1e12, neginf=-1e12))
        val = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        left = np.searchsorted(ref, val, side="left")
        right = np.searchsorted(ref, val, side="right")
        return (left + right) / (2.0 * max(len(ref), 1))

    e_pct = percentile_from_train(train_energy, eval_energy)
    d_pct = percentile_from_train(train_dist, eval_dist)
    return 0.5 * e_pct + 0.5 * d_pct
