import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "A principal encoder subspace captures common in-scope variation, while the largest coordinate-level deviation left in its residual can expose sparse unsupported feature shifts that a residual L2 norm dilutes."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004", "m_0042"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)

    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]

    total = vals.sum()
    if total > 0.0:
        keep = int(np.searchsorted(np.cumsum(vals), 0.90 * total) + 1)
    else:
        keep = 1
    keep = min(max(keep, 1), x.shape[1] - 1)

    basis = vecs[:, :keep]
    train_resid = xc - (xc @ basis) @ basis.T
    eval_centered = z - mean
    eval_resid = eval_centered - (eval_centered @ basis) @ basis.T

    scale = np.sqrt(np.mean(train_resid * train_resid, axis=0))
    floor = np.median(scale[scale > 0.0]) if np.any(scale > 0.0) else 1.0
    scale = np.maximum(scale, max(floor * 1e-3, 1e-8))

    return np.max(np.abs(eval_resid) / scale, axis=1)
