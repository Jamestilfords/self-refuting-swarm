import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier prediction that disagrees with the nearest class under class-specific "
    "shrunk diagonal variances indicates feature-to-logit inconsistency that is uncommon in scope."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0037", "m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    sq_sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sq_sums, y, centered * centered)
    class_var = sq_sums / np.maximum(counts[:, None], 1.0)

    global_var = np.mean(centered * centered, axis=0)
    shrink = 24.0
    variances = (
        counts[:, None] * class_var + shrink * global_var[None, :]
    ) / (counts[:, None] + shrink)
    floor = np.maximum(np.median(global_var) * 1e-6, 1e-10)
    variances = np.maximum(variances, floor)

    pred = np.argmax(ev.logits, axis=1)
    selected = np.empty(z.shape[0], dtype=np.float64)
    nearest = np.full(z.shape[0], np.inf, dtype=np.float64)

    for c in range(k):
        delta = z - means[c]
        dist = np.sum(delta * delta / variances[c], axis=1)
        nearest = np.minimum(nearest, dist)
        mask = pred == c
        if np.any(mask):
            selected[mask] = dist[mask]

    return selected - nearest
