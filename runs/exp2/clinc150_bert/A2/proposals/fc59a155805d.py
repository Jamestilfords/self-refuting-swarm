import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "The mean distance to several nearby training queries estimates local in-scope support more "
    "robustly than a single neighbor. Fine-tuned bottleneck directions should cluster by known intent, "
    "while OOS queries tend to have weak multi-neighbor support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0091"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    # kNN helper returns the kth-neighbor distance; average several neighborhood scales.
    distances = []
    for k in (5, 15, 35):
        distances.append(ctx.h.knn(x_train, x_eval, k=k, l2=False))
    return np.mean(np.stack(distances, axis=1), axis=1)
