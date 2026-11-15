import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "OOS queries tend to require a larger expansion from their nearest semantic neighbor "
    "to a broader local neighborhood, reflecting weak or fragmented in-scope support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0067", "m_0071"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    d_near_train = ctx.h.knn(tr.pooled, tr.pooled, k=6, l2=True)
    d_wide_train = ctx.h.knn(tr.pooled, tr.pooled, k=33, l2=True)
    train_score = d_wide_train / (d_near_train + 1e-6)

    d_near_eval = ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True)
    d_wide_eval = ctx.h.knn(tr.pooled, ev.pooled, k=32, l2=True)
    raw = d_wide_eval / (d_near_eval + 1e-6)

    lo, hi = np.quantile(train_score, [0.02, 0.98])
    scale = max(hi - lo, 1e-6)
    return np.clip((raw - lo) / scale, -10.0, 10.0)
