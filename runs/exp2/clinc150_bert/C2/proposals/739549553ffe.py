import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and class-conditional feature distance capture complementary uncertainty signals. Mapping each to its training empirical percentile puts them on a common scale without using evaluation-set statistics."
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

    e_sorted = np.sort(np.nan_to_num(train_energy, nan=0.0, posinf=1e12, neginf=-1e12))
    d_sorted = np.sort(np.nan_to_num(train_dist, nan=0.0, posinf=1e12, neginf=-1e12))

    eval_energy = np.nan_to_num(eval_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e12, neginf=-1e12)

    e_pct = np.searchsorted(e_sorted, eval_energy, side="right") / float(len(e_sorted))
    d_pct = np.searchsorted(d_sorted, eval_dist, side="right") / float(len(d_sorted))

    return 0.5 * e_pct + 0.5 * d_pct
