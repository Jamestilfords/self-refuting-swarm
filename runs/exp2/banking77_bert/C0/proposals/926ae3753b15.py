import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries may show unusually large deviations along low-variance within-class directions, especially relative to the class selected by the classifier. A predicted-class trailing-subspace Mahalanobis residual preserves this signal without collapsing it to the previous nearest-class bottom-space score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    means = np.zeros((k, x.shape[1]), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = centered.T @ centered / max(len(x) - k, 1)
    vals, vecs = np.linalg.eigh(cov)

    positive = vals[vals > 1e-12]
    scale = np.median(positive) if positive.size else 1.0
    tail_dim = min(160, x.shape[1])
    tail_vals = np.maximum(vals[:tail_dim], 1e-4 * scale)
    tail_vecs = vecs[:, :tail_dim]

    pred = np.argmax(ev.logits, axis=1)
    residual = z - means[pred]
    projected = residual @ tail_vecs
    scores = np.sum((projected * projected) / tail_vals[None, :], axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
