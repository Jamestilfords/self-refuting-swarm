import numpy as np
from scipy.spatial import cKDTree

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode constraints that in-scope examples rarely violate. Class-conditional local support in a whitened trailing-eigenspace detects queries whose subtle residual variation is unsupported by their predicted intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    pred = np.argmax(ev.logits, axis=1).astype(np.int64)

    d = x.shape[1]
    n_tail = min(64, max(8, d // 8))
    classes = np.unique(y)

    means = np.zeros((int(max(classes.max(), pred.max())) + 1, d), dtype=np.float64)
    for c in classes:
        means[c] = x[y == c].mean(axis=0)

    residual = x - means[y]
    cov = (residual.T @ residual) / max(len(x) - len(classes), 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[:n_tail]
    basis = vecs[:, order]
    scale = np.sqrt(np.maximum(vals[order], 1e-8))

    tx = (residual @ basis) / scale
    ez = (z - means[np.minimum(pred, means.shape[0] - 1)]) @ basis / scale

    global_tree = cKDTree(tx)
    out = np.empty(len(z), dtype=np.float64)

    for c in np.unique(pred):
        idx = np.where(pred == c)[0]
        train_idx = np.where(y == c)[0]
        if len(train_idx) >= 2 and c < means.shape[0]:
            tree = cKDTree(tx[train_idx])
            k = min(5, len(train_idx))
        else:
            tree = global_tree
            k = min(5, len(tx))
        dist, _ = tree.query(ez[idx], k=k)
        if k > 1:
            dist = np.asarray(dist)[:, -1]
        out[idx] = np.asarray(dist, dtype=np.float64)

    out = np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
    return out + 1e-12 * ctx.h.energy(ev.logits)
