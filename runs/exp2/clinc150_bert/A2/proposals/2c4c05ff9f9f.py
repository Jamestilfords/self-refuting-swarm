import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Known intents are tightly constrained along trailing within-class covariance directions, "
    "so whitened residual energy in those low-variance directions exposes off-manifold OOS queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0031"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    dim = x.shape[1]
    counts = np.bincount(y, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, dim), dtype=float)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - n_classes, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    tail_dim = min(64, max(8, dim // 4))
    tail_vals = vals[:tail_dim]
    tail_vecs = vecs[:, :tail_dim]

    positive = vals[vals > 1e-10]
    reference = np.median(positive) if positive.size else 1.0
    safe_vals = np.maximum(tail_vals, max(reference * 1e-4, 1e-8))

    predicted = np.argmax(ev.logits, axis=1)
    centered = z - means[predicted]
    projected = centered @ tail_vecs
    scores = np.sum((projected * projected) / safe_vals[None, :], axis=1)
    return np.clip(scores, 0.0, 1e12)
