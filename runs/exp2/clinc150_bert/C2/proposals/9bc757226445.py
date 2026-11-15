import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode constraints that in-scope examples rarely violate. This score uses the classifier-predicted class and flags unusually large sparse deviations in the trailing whitened eigenspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)

    k = tr.logits.shape[1]
    d = x.shape[1]
    counts = np.bincount(y, minlength=k).astype(float)
    means = np.zeros((k, d), dtype=float)
    np.add.at(means, y, x)
    global_mean = np.mean(x, axis=0)
    valid = counts > 0
    means[valid] /= counts[valid, None]
    means[~valid] = global_mean

    centered = x - means[y]
    denom = max(x.shape[0] - int(np.sum(valid)), 1)
    cov = (centered.T @ centered) / float(denom)
    vals, vecs = np.linalg.eigh(cov)

    tail_dim = min(96, d)
    tail_vals = vals[:tail_dim]
    tail_vecs = vecs[:, :tail_dim]
    floor = max(float(np.max(vals)) * 1e-6, 1e-10)
    inv_scale = 1.0 / np.sqrt(np.maximum(tail_vals, floor))

    pred = np.argmax(ev.logits, axis=1)
    residual = z - means[pred]
    tail = (residual @ tail_vecs) * inv_scale
    absolute = np.abs(tail)

    rank = min(4, tail_dim)
    sparse_violation = np.partition(absolute, tail_dim - rank, axis=1)[:, tail_dim - rank]
    return np.nan_to_num(sparse_violation, nan=0.0, posinf=1e6, neginf=0.0)
