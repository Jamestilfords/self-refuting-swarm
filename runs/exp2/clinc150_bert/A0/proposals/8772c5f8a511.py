import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that supported intents tend to satisfy. "
    "A class-conditioned, ridge-whitened residual in these trailing directions flags inputs that violate those constraints."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0091"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.feats, dtype=np.float64)

    n, d = x.shape
    k_classes = tr.logits.shape[1]
    counts = np.bincount(y, minlength=k_classes).astype(np.float64)

    global_mean = x.mean(axis=0)
    sums = np.zeros((k_classes, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = np.tile(global_mean, (k_classes, 1))
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - k_classes, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    tail_dim = min(32, d)
    tail_vals = eigvals[:tail_dim]
    tail_vecs = eigvecs[:, :tail_dim]

    positive = eigvals[eigvals > 0]
    base = np.median(positive) if positive.size else 1.0
    ridge = max(base * 1e-3, 1e-8)
    precision = 1.0 / np.maximum(tail_vals, ridge)

    train_tail = centered @ tail_vecs
    train_energy = np.sum(train_tail * train_tail * precision, axis=1)

    class_scale = np.full(k_classes, np.median(train_energy) + 1e-8, dtype=np.float64)
    for c in np.flatnonzero(present):
        values = train_energy[y == c]
        scale = np.median(values)
        if np.isfinite(scale) and scale > 1e-10:
            class_scale[c] = scale

    pred = np.argmax(ev.logits, axis=1)
    eval_centered = z - means[pred]
    eval_tail = eval_centered @ tail_vecs
    eval_energy = np.sum(eval_tail * eval_tail * precision, axis=1)

    return np.log1p(eval_energy / class_scale[pred])
