import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should remain close to the class selected by the classifier in low-variance within-class directions, whereas OOS inputs often violate these tightly constrained residual features. Conditioning the trailing-subspace residual on the predicted class makes this distinct from a nearest-class GDA score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)

    n, d = x.shape
    classes = int(np.max(y)) + 1
    means = np.zeros((classes, d), dtype=float)
    counts = np.bincount(y, minlength=classes).astype(float)

    for c in range(classes):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - classes, 1)
    vals, vecs = np.linalg.eigh(cov)

    tail_dim = min(96, max(16, d // 8))
    tail_vecs = vecs[:, :tail_dim]
    tail_vals = vals[:tail_dim]
    floor = max(float(np.mean(np.maximum(vals, 0.0))) * 1e-6, 1e-10)
    scales = np.sqrt(np.maximum(tail_vals, floor))

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, classes - 1)
    residual = (z - means[pred]) @ tail_vecs
    scores = np.sum((residual / scales) ** 2, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
