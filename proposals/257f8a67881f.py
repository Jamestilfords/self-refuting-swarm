import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident queries retain the training-calibrated energy ordering, while only energy-uncertain "
    "queries are reranked by normalized pooled-feature Mahalanobis support."
)
SOURCE = "https://proceedings.mlr.press/v267/muller25a.html"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0079"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    threshold = float(np.quantile(train_energy, 0.65))
    e_sorted = np.sort(train_energy)
    e_rank = np.searchsorted(e_sorted, eval_energy, side="right")
    e_tail = e_rank / float(len(e_sorted) + 1)

    train_norm = ctx.h.l2(tr.pooled)
    eval_norm = ctx.h.l2(ev.pooled)
    g = ctx.h.gda(train_norm, tr.labels, cov="tied", whiten=True)

    train_dist = g.min_mahalanobis_sq(train_norm)
    eval_dist = g.min_mahalanobis_sq(eval_norm)
    d_sorted = np.sort(train_dist)
    d_rank = np.searchsorted(d_sorted, eval_dist, side="right")
    d_tail = d_rank / float(len(d_sorted) + 1)

    q = np.searchsorted(e_sorted, threshold, side="right") / float(len(e_sorted) + 1)
    q = min(max(q, eps), 1.0 - eps)

    confident = eval_energy <= threshold
    scores = np.where(
        confident,
        e_tail,
        q + (1.0 - q) * d_tail,
    )
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
