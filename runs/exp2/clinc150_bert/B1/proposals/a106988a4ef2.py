import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that ordinary intent examples satisfy. "
    "This score measures the class-conditional breadth and depth of empirical tail violations across those trailing directions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0135"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)
    means = np.tile(global_mean, (k, 1))
    counts = np.bincount(y, minlength=k)

    for c in range(k):
        if counts[c] > 0:
            means[c] = np.mean(x[y == c], axis=0)

    centered = x - means[np.clip(y, 0, k - 1)]
    cov = (centered.T @ centered) / max(n - k, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    r = min(128, max(16, d // 4))
    vals = vals[:r]
    vecs = vecs[:, :r]

    positive = vals[vals > 1e-12]
    floor = (np.median(positive) if positive.size else 1.0) * 1e-4
    scale = np.sqrt(np.maximum(vals, floor))

    train_proj = np.abs((centered @ vecs) / scale)
    global_q = np.quantile(train_proj, 0.95, axis=0)
    thresholds = np.tile(global_q, (k, 1))

    for c in range(k):
        if counts[c] >= 8:
            thresholds[c] = np.quantile(train_proj[y == c], 0.95, axis=0)

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, k - 1)
    eval_centered = z - means[pred]
    eval_proj = np.abs((eval_centered @ vecs) / scale)

    q = np.maximum(thresholds[pred], 1e-6)
    excess = np.maximum(eval_proj - q, 0.0) / q

    breadth = np.mean(excess > 0.0, axis=1)
    depth = np.mean(np.log1p(excess), axis=1)
    continuity = 1e-3 * np.mean(eval_proj / q, axis=1)
    out = breadth + depth + continuity

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
