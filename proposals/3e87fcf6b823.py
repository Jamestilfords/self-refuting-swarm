import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions are tightly constrained for supported intents. "
    "Taking the minimum class-wise sum of the largest standardized trailing-direction residuals emphasizes sparse OOS violations that ordinary Mahalanobis distance can dilute."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0006", "m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.feats, dtype=np.float64)

    classes, inv = np.unique(y, return_inverse=True)
    n, d = x.shape
    c = len(classes)

    means = np.zeros((c, d), dtype=np.float64)
    np.add.at(means, inv, x)
    counts = np.bincount(inv, minlength=c).astype(np.float64)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[inv]
    denom = max(n - c, 1)
    cov = (centered.T @ centered) / denom
    ridge = max(float(np.trace(cov)) / max(d, 1), 1e-8) * 1e-4
    cov.flat[:: d + 1] += ridge

    vals, vecs = np.linalg.eigh(cov)
    tail_dim = min(64, d)
    tail_vals = np.maximum(vals[:tail_dim], ridge)
    tail_vecs = vecs[:, :tail_dim]
    scale = 1.0 / np.sqrt(tail_vals)

    q = min(12, tail_dim)
    out = np.empty(z.shape[0], dtype=np.float64)
    chunk = 256

    for start in range(0, z.shape[0], chunk):
        stop = min(start + chunk, z.shape[0])
        zb = z[start:stop]
        best = np.full(stop - start, np.inf, dtype=np.float64)

        for j in range(c):
            residual = (zb - means[j]) @ tail_vecs
            sq = (residual * scale) ** 2
            if q < tail_dim:
                largest = np.partition(sq, tail_dim - q, axis=1)[:, tail_dim - q:]
                value = np.sum(largest, axis=1)
            else:
                value = np.sum(sq, axis=1)
            best = np.minimum(best, value)

        out[start:stop] = best

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
