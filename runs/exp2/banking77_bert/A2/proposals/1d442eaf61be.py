import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-intent directions encode tight representation constraints. "
    "Rather than summing residual energy, this score detects the largest standardized trailing-direction violation for the classifier-predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0097"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    global_mean = x.mean(axis=0)
    means = sums / np.maximum(counts[:, None], 1.0)
    means[counts == 0] = global_mean

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - k, 1)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)

    scale = max(float(np.max(vals)), 1e-12)
    usable = np.flatnonzero(vals > scale * 1e-8)
    if usable.size:
        take = usable[:min(64, usable.size)]
    else:
        take = np.array([int(np.argmax(vals))], dtype=np.int64)
    basis = vecs[:, take]

    train_proj = residuals @ basis
    q = train_proj.shape[1]
    class_std = np.empty((k, q), dtype=np.float64)
    class_center = np.empty(k, dtype=np.float64)
    class_spread = np.empty(k, dtype=np.float64)

    raw_train = np.empty(len(x), dtype=np.float64)
    for c in range(k):
        idx = y == c
        if np.any(idx):
            pc = train_proj[idx]
            sd = np.sqrt(np.mean(pc * pc, axis=0))
            floor = max(float(np.median(sd)) * 1e-3, 1e-8)
            class_std[c] = np.maximum(sd, floor)
            violations = np.max(np.abs(pc / class_std[c]), axis=1)
            med = float(np.median(violations))
            mad = float(np.median(np.abs(violations - med)))
            class_center[c] = med
            class_spread[c] = max(1.4826 * mad, 0.05)
            raw_train[idx] = violations
        else:
            class_std[c] = 1.0
            class_center[c] = 0.0
            class_spread[c] = 1.0

    pred = np.argmax(ev.logits, axis=1)
    eval_proj = (z - means[pred]) @ basis
    violations = np.max(np.abs(eval_proj) / class_std[pred], axis=1)
    scores = (violations - class_center[pred]) / class_spread[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
