import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "The low-variance within-class subspace contains directions that in-scope examples are tightly constrained to follow. Measuring trailing-subspace deviation from the classifier-predicted class mean makes this signal class-conditional and avoids the ranking of a nearest-class trailing residual."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)

    n_classes = int(np.max(y)) + 1
    means = np.zeros((n_classes, x.shape[1]), dtype=float)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - n_classes, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    n_trailing = min(64, x.shape[1])
    basis = vecs[:, order[:n_trailing]]

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    pred = np.clip(pred, 0, n_classes - 1)
    delta = (z - means[pred]) @ basis
    scores = np.sum(delta * delta, axis=1)

    scale = np.median(np.sum((residuals @ basis) ** 2, axis=1))
    scores = scores / max(float(scale), 1e-12)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
