import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions are tightly constrained for known intents, and a "
    "class-conditioned fourth-moment residual emphasizes sparse violations that a summed "
    "Mahalanobis radius can dilute."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0188", "m_0185", "m_0195"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z_eval = np.asarray(ev.feats, dtype=float)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)

    counts = np.bincount(y, minlength=k).astype(float)
    sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    means = np.repeat(global_mean[None, :], k, axis=0)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - np.sum(present), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    scale = float(np.median(np.abs(eigvals)))
    floor = max(scale * 1e-6, 1e-8)
    eigvals = np.maximum(eigvals, floor)

    q = min(64, d)
    basis = eigvecs[:, :q]
    inv_std = 1.0 / np.sqrt(eigvals[:q])

    train_trailing = (centered @ basis) * inv_std
    train_trailing = np.clip(train_trailing, -1e6, 1e6)
    train_l4 = np.mean(train_trailing ** 4, axis=1)

    class_median = np.full(k, np.median(train_l4) + 1e-8, dtype=float)
    for c in range(k):
        if counts[c] > 0:
            class_median[c] = np.median(train_l4[y == c]) + 1e-8

    pred = np.argmax(ev.logits, axis=1)
    eval_centered = z_eval - means[pred]
    eval_trailing = (eval_centered @ basis) * inv_std
    eval_trailing = np.clip(eval_trailing, -1e6, 1e6)
    eval_l4 = np.mean(eval_trailing ** 4, axis=1)

    return np.log1p(eval_l4 / class_median[pred])
