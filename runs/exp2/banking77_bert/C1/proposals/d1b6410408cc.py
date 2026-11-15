import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode features that in-scope examples are tightly constrained to preserve. Measuring tail-spectrum deviation from the classifier's predicted class mean avoids the minimum-over-classes ranking that duplicated the earlier partial Mahalanobis detector."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=float)

    classes, yi = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    dim = x.shape[1]

    counts = np.bincount(yi, minlength=n_classes).astype(float)
    means = np.zeros((n_classes, dim), dtype=float)
    np.add.at(means, yi, x)
    means /= counts[:, None]

    centered = x - means[yi]
    cov = centered.T @ centered / max(len(x) - n_classes, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    tail_dim = min(96, dim)
    basis = eigvecs[:, :tail_dim]
    scale = np.maximum(eigvals[:tail_dim], np.median(eigvals) * 1e-4 + 1e-10)

    pred_labels = np.argmax(ev.logits, axis=1)
    lookup = {int(label): i for i, label in enumerate(classes)}
    pred_idx = np.array([lookup.get(int(label), 0) for label in pred_labels])

    residual = (z - means[pred_idx]) @ basis
    return np.sum((residual * residual) / scale[None, :], axis=1)
