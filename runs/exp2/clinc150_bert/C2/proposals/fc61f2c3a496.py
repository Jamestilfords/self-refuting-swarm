import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the score for training-like confident predictions, while a train-calibrated gate increasingly substitutes pooled-feature Mahalanobis distance when energy indicates uncertainty. This can catch OOS inputs whose logits are only moderately uncertain but whose encoder representations are atypical."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    e_sorted = np.sort(np.asarray(train_e, dtype=float))
    d_sorted = np.sort(np.asarray(train_d, dtype=float))
    n = max(len(e_sorted), 1)

    e_rank = np.searchsorted(e_sorted, eval_e, side="right") / n
    d_rank = np.searchsorted(d_sorted, eval_d, side="right") / n

    q25, q70, q75 = np.quantile(train_e, [0.25, 0.70, 0.75])
    scale = max(float(q75 - q25), 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_e - q70) / scale, -40.0, 40.0)))

    scores = (1.0 - gate) * e_rank + gate * d_rank
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
