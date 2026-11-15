import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions are tightly constrained for in-scope examples, so displacement from the classifier's predicted-class centroid in this subspace is informative for OOS detection. Using the predicted class rather than a nearest-centroid minimum retains disagreement between representation and classifier evidence."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - k, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    n_tail = min(96, max(16, d // 8))
    tail_vals = vals[:n_tail]
    tail_vecs = vecs[:, :n_tail]

    positive = vals[vals > 1e-12]
    floor = (np.median(positive) * 1e-4) if positive.size else 1e-6
    scales = 1.0 / np.maximum(tail_vals, floor)

    pred = np.argmax(ev.logits, axis=1)
    delta = np.asarray(ev.pooled, dtype=np.float64) - means[pred]
    projected = delta @ tail_vecs
    scores = np.sum(projected * projected * scales[None, :], axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
