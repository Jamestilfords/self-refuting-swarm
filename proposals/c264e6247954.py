import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions are tightly constrained, so an OOS query often has unusually large trailing-space residual energy. "
    "Its whitened trailing residual can also have an atypical concentration pattern across directions, distinguishing coherent unsupported shifts from ordinary in-class variation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0006", "m_0137"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.feats, dtype=np.float64)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = np.repeat(global_mean[None, :], k, axis=0)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    residuals = x - means[y]
    denom = max(n - int(np.sum(present)), 1)
    cov = (residuals.T @ residuals) / denom
    cov.flat[:: d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)

    eigvals, eigvecs = np.linalg.eigh(cov)
    q = min(48, d)
    vals = eigvals[:q]
    vecs = eigvecs[:, :q]

    positive = eigvals[eigvals > 1e-12]
    floor = max(float(np.percentile(positive, 5)) * 1e-3, 1e-8) if positive.size else 1e-8
    inv_scale = 1.0 / np.sqrt(np.maximum(vals, floor))

    train_tail = (residuals @ vecs) * inv_scale
    train_r2 = np.sum(train_tail * train_tail, axis=1)
    train_l1 = np.sum(np.abs(train_tail), axis=1)
    train_shape = (train_l1 * train_l1) / (q * train_r2 + 1e-12)

    global_r_med = np.median(train_r2)
    global_r_mad = np.median(np.abs(train_r2 - global_r_med)) + 1e-6
    global_s_med = np.median(train_shape)
    global_s_mad = np.median(np.abs(train_shape - global_s_med)) + 1e-6

    r_med = np.full(k, global_r_med, dtype=np.float64)
    r_mad = np.full(k, global_r_mad, dtype=np.float64)
    s_med = np.full(k, global_s_med, dtype=np.float64)
    s_mad = np.full(k, global_s_mad, dtype=np.float64)

    for c in range(k):
        idx = y == c
        if np.sum(idx) >= 8:
            rc = train_r2[idx]
            sc = train_shape[idx]
            r_med[c] = np.median(rc)
            r_mad[c] = np.median(np.abs(rc - r_med[c])) + 1e-6
            s_med[c] = np.median(sc)
            s_mad[c] = np.median(np.abs(sc - s_med[c])) + 1e-6

    pred = np.argmax(ev.logits, axis=1)
    eval_tail = (z - means[pred]) @ vecs
    eval_tail *= inv_scale
    r2 = np.sum(eval_tail * eval_tail, axis=1)
    l1 = np.sum(np.abs(eval_tail), axis=1)
    shape = (l1 * l1) / (q * r2 + 1e-12)

    radial_excess = np.maximum((r2 - r_med[pred]) / (1.4826 * r_mad[pred]), 0.0)
    shape_deviation = np.abs(shape - s_med[pred]) / (1.4826 * s_mad[pred])
    scores = np.clip(radial_excess, 0.0, 1000.0) + 0.35 * np.clip(shape_deviation, 0.0, 1000.0)
    return scores
