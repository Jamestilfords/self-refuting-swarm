import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions expose small residual shifts that ordinary semantic variation can hide. Calibrating the trailing-subspace residual against the predicted class prevents broad in-scope classes from dominating the score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]
    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(n - k, 1)
    ridge = max(float(np.trace(cov)) / max(d, 1), 1e-8) * 1e-5
    vals, vecs = np.linalg.eigh(cov + ridge * np.eye(d))

    qdim = min(96, d)
    vals = np.maximum(vals[:qdim], ridge)
    basis = vecs[:, :qdim]

    train_proj = (residuals @ basis) / np.sqrt(vals)
    train_q = np.sum(train_proj * train_proj, axis=1)

    global_center = float(np.median(train_q))
    global_scale = float(np.median(np.abs(train_q - global_center)) * 1.4826)
    global_scale = max(global_scale, 1e-6)

    centers = np.full(k, global_center, dtype=float)
    scales = np.full(k, global_scale, dtype=float)
    for c in range(k):
        qc = train_q[y == c]
        if qc.size >= 5:
            med = float(np.median(qc))
            mad = float(np.median(np.abs(qc - med)) * 1.4826)
            centers[c] = med
            scales[c] = max(mad, global_scale * 0.1, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    eval_residuals = z - means[pred]
    eval_proj = (eval_residuals @ basis) / np.sqrt(vals)
    eval_q = np.sum(eval_proj * eval_proj, axis=1)
    scores = (eval_q - centers[pred]) / scales[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
