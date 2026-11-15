import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode constraints that supported intent representations should satisfy. A predicted-class conditional residual score emphasizes violations of these constrained directions while accounting for intent-specific feature centers."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0053"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    global_mean = x.mean(axis=0)
    valid = counts > 0
    means[valid] /= counts[valid, None]
    means[~valid] = global_mean

    residuals = x - means[y]
    cov = residuals.T @ residuals / max(x.shape[0] - int(valid.sum()), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    n_tail = max(16, d // 5)
    tail_vals = eigvals[:n_tail]
    tail_vecs = eigvecs[:, :n_tail]
    positive = eigvals[eigvals > 0]
    scale = np.median(positive) if positive.size else 1.0
    denom = np.maximum(tail_vals, scale * 1e-3)

    pred = np.argmax(ev.logits, axis=1)
    test_residuals = np.asarray(ev.pooled, dtype=np.float64) - means[pred]
    projected = test_residuals @ tail_vecs
    scores = np.sum((projected * projected) / denom[None, :], axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
