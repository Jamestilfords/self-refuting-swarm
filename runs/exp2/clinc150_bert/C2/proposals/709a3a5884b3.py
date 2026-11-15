import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode constraints that in-scope utterances should satisfy. Measuring residuals in these directions around the classifier-predicted class makes the trailing-space signal class-conditional rather than a global nearest-class ranking."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    n, d = x.shape
    k = tr.logits.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - k, 1)
    vals, vecs = np.linalg.eigh(cov)

    q = min(64, d)
    low_vals = np.maximum(vals[:q], 1e-6)
    low_vecs = vecs[:, :q]

    pred = np.argmax(ev.logits, axis=1)
    residual = z - means[pred]
    coords = residual @ low_vecs
    scores = np.sum((coords * coords) / low_vals[None, :], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
