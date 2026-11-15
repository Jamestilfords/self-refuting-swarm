import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Within-intent low-variance directions represent feature changes that ID training data rarely exhibits. Measuring class-centered residual energy in this trailing subspace should expose OOS queries even when they remain close to an intent in dominant feature directions."
SOURCE = "https://doi.org/10.48550/arxiv.2203.10807"
PARENTS = []
MEMORY_USED = ["m_0044", "m_0045", "m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    global_mean = x.mean(axis=0)
    present = counts > 0
    means[present] /= counts[present, None]
    means[~present] = global_mean

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - k, 1)
    ridge = (np.trace(cov) / max(d, 1) + 1e-8) * 1e-6
    eigvals, eigvecs = np.linalg.eigh(cov + ridge * np.eye(d))

    n_trailing = min(128, max(16, d // 4))
    trailing = eigvecs[:, :n_trailing]
    predicted = np.argmax(ev.logits, axis=1)
    residual = np.asarray(ev.pooled, dtype=np.float64) - means[predicted]
    projected = residual @ trailing

    scale = np.maximum(eigvals[:n_trailing], ridge)
    scores = np.sum((projected * projected) / scale[None, :], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
