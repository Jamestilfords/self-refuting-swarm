import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode constraints that ordinary semantic variation rarely violates. Measuring a query against the trailing-subspace residual distribution of its classifier-predicted intent highlights encoder/classifier disagreement without requiring OOS calibration data."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes = int(max(np.max(y), tr.logits.shape[1] - 1) + 1)
    dim = x.shape[1]
    tail_dim = min(96, max(16, dim // 8))

    means = np.zeros((classes, dim), dtype=float)
    counts = np.bincount(y, minlength=classes).astype(float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    residual = x - means[y]
    cov = residual.T @ residual / max(x.shape[0] - classes, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, max(float(np.trace(cov)) / max(dim, 1) * 1e-5, 1e-8))
    basis = vecs[:, :tail_dim]
    inv_scale = 1.0 / vals[:tail_dim]

    train_proj = residual @ basis
    train_tail = np.sum(train_proj * train_proj * inv_scale[None, :], axis=1)

    med = np.zeros(classes, dtype=float)
    mad = np.ones(classes, dtype=float)
    global_med = float(np.median(train_tail))
    global_mad = float(np.median(np.abs(train_tail - global_med)) * 1.4826)
    global_mad = max(global_mad, 1e-6)

    for c in range(classes):
        values = train_tail[y == c]
        if values.size >= 4:
            m = float(np.median(values))
            s = float(np.median(np.abs(values - m)) * 1.4826)
            med[c] = m
            mad[c] = max(s, 1e-6)
        else:
            med[c] = global_med
            mad[c] = global_mad

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, classes - 1)
    eval_residual = z - means[pred]
    eval_proj = eval_residual @ basis
    eval_tail = np.sum(eval_proj * eval_proj * inv_scale[None, :], axis=1)

    out = (eval_tail - med[pred]) / mad[pred]
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
