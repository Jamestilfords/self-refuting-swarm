import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS representations can produce unusually large standardized deviations in individual low-variance within-class directions. Using an extreme-value statistic over trailing directions differs from residual-energy scoring by emphasizing sparse violations of the learned class manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - k, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    tail_dim = min(64, max(16, d // 8))
    tail_vals = vals[:tail_dim]
    tail_vecs = vecs[:, :tail_dim]

    floor = max(float(np.median(vals)) * 1e-5, 1e-8)
    scale = np.sqrt(np.maximum(tail_vals, floor))

    pred = np.argmax(ev.logits, axis=1)
    residual = np.asarray(ev.pooled, dtype=float) - means[pred]
    z = np.abs((residual @ tail_vecs) / scale)

    largest = min(5, tail_dim)
    return np.sum(np.partition(z, tail_dim - largest, axis=1)[:, -largest:], axis=1)
