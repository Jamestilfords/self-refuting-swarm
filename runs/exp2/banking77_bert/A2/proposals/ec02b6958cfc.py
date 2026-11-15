import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that in-scope representations "
    "should satisfy. A class-calibrated whitened deviation in this trailing subspace detects "
    "queries that violate those constraints without relying on local neighbor support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0075"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)

    n, d = x.shape
    k = tr.logits.shape[1]
    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)

    for c in range(k):
        mask = y == c
        if np.any(mask):
            means[c] = x[mask].mean(axis=0)
        else:
            means[c] = x.mean(axis=0)

    centered = x - means[y]
    denom = max(n - np.count_nonzero(counts), 1)
    cov = (centered.T @ centered) / float(denom)
    cov = 0.5 * (cov + cov.T)

    diag_scale = float(np.mean(np.diag(cov)))
    ridge = max(diag_scale * 1e-5, 1e-8)
    vals, vecs = np.linalg.eigh(cov + ridge * np.eye(d))

    r = min(128, d)
    vals = np.maximum(vals[:r], ridge)
    basis = vecs[:, :r]

    train_proj = centered @ basis
    train_dist = np.sum((train_proj * train_proj) / vals, axis=1)

    class_mean = np.zeros(k, dtype=float)
    class_std = np.ones(k, dtype=float)
    global_mean = float(np.mean(train_dist))
    global_std = max(float(np.std(train_dist)), 1e-6)

    for c in range(k):
        vals_c = train_dist[y == c]
        if vals_c.size >= 2:
            class_mean[c] = float(np.mean(vals_c))
            class_std[c] = max(float(np.std(vals_c)), 1e-6)
        else:
            class_mean[c] = global_mean
            class_std[c] = global_std

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, k - 1)
    ev_centered = z - means[pred]
    ev_proj = ev_centered @ basis
    ev_dist = np.sum((ev_proj * ev_proj) / vals, axis=1)

    scores = (ev_dist - class_mean[pred]) / class_std[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
