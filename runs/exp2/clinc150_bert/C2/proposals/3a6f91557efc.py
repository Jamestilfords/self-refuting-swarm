import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries can remain close along dominant semantic directions while violating low-variance within-class constraints. This uses the predicted intent's centroid and a shrinkage-whitened trailing within-class eigenspace, rather than the generic nearest-class trailing score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)

    n_classes = tr.logits.shape[1]
    d = x.shape[1]
    means = np.zeros((n_classes, d), dtype=float)
    counts = np.bincount(y, minlength=n_classes).astype(float)

    for c in range(n_classes):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)

    residuals = x - means[y]
    cov = residuals.T @ residuals / max(len(x) - n_classes, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    tail_dim = min(96, d)
    tail_vals = vals[:tail_dim]
    tail_vecs = vecs[:, :tail_dim]

    positive = vals[vals > 1e-12]
    floor = (np.median(positive) * 1e-3) if positive.size else 1e-6
    scales = np.maximum(tail_vals, max(floor, 1e-8))

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, n_classes - 1)
    centered = z - means[predicted]
    projected = centered @ tail_vecs

    scores = np.sum((projected * projected) / scales, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
