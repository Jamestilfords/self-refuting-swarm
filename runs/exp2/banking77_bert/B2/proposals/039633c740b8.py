import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Within-class low-variance directions are tightly constrained for known intents. "
    "The minimum class-conditional residual energy in these trailing directions flags inputs unsupported by every known intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0035"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    classes = np.unique(y)
    d = x.shape[1]
    means = np.empty((classes.size, d), dtype=np.float64)
    residuals = np.empty_like(x)

    for i, c in enumerate(classes):
        mask = y == c
        means[i] = x[mask].mean(axis=0)
        residuals[mask] = x[mask] - means[i]

    cov = (residuals.T @ residuals) / max(x.shape[0] - classes.size, 1)
    cov.flat[:: d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)
    vals, vecs = np.linalg.eigh(cov)

    q = max(8, min(64, d // 4))
    basis = vecs[:, :q]
    scale = np.maximum(vals[:q], 1e-8 * np.maximum(vals[-1], 1.0))

    best = np.full(z.shape[0], np.inf, dtype=np.float64)
    for mean in means:
        projected = (z - mean) @ basis
        dist = np.sum((projected * projected) / scale, axis=1)
        best = np.minimum(best, dist)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=0.0)
