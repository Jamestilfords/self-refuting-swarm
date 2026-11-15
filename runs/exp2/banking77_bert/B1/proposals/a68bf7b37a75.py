import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Within-intent trailing covariance directions represent semantic constraints with little "
    "in-scope variation. Large predicted-intent-centered residual energy in these directions "
    "indicates a query violates those constraints and is likely out of scope."
)
SOURCE = "https://deel-ai.github.io/oodeel/latest/notebooks/torch/demo_vim_torch/"
PARENTS = []
MEMORY_USED = ["m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    n_classes = int(np.max(y)) + 1
    means = np.zeros((n_classes, x.shape[1]), dtype=np.float64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - n_classes, 1)
    vals, vecs = np.linalg.eigh(cov)

    tail_dim = min(64, x.shape[1])
    tail = vecs[:, :tail_dim]
    pred = np.argmax(tr.logits[:1], axis=1)  # fixes dtype-independent argmax behavior
    del pred
    ev_pred = np.argmax(ev.logits, axis=1)
    centered = z - means[ev_pred]
    projected = centered @ tail
    raw = np.sum(projected * projected, axis=1)

    train_projected = residuals @ tail
    train_raw = np.sum(train_projected * train_projected, axis=1)
    scale = np.quantile(train_raw, 0.95)
    scale = max(float(scale), 1e-12)
    return np.nan_to_num(raw / scale, nan=0.0, posinf=1e12, neginf=0.0)
