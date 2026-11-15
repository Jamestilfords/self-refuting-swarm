import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-conditional residuals along trailing within-class covariance directions measure violations of low-variance semantic constraints. Conditioning the residual on the classifier's predicted class avoids reproducing a global trailing-direction distance."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]
    counts = np.bincount(y, minlength=k).astype(float)
    means = np.zeros((k, d), dtype=float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    global_mean = x.mean(axis=0)
    missing = counts == 0
    if np.any(missing):
        means[missing] = global_mean

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - k, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    scale = max(float(np.mean(vals)), 1e-8)
    ridge = 1e-4 * scale

    stable = np.flatnonzero(vals > 1e-7 * scale)
    if stable.size:
        tail_idx = stable[:min(48, stable.size)]
    else:
        tail_idx = np.arange(min(48, d))

    basis = vecs[:, tail_idx]
    denom = vals[tail_idx] + ridge

    pred = np.argmax(ev.logits, axis=1)
    residual = z - means[pred]
    projection = residual @ basis
    tail_energy = np.sum((projection * projection) / denom[None, :], axis=1)

    return np.nan_to_num(
        np.log1p(tail_energy),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )
