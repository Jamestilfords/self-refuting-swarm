import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Trailing shared within-class covariance directions encode constraints that in-scope examples "
    "rarely violate. Calibrating this residual energy separately for the classifier-predicted intent "
    "avoids a globally monotone duplicate of an unconditioned trailing-eigenspace score."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0075"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - k, 1)
    ridge = 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov + ridge * np.eye(d))

    n_tail = min(96, max(8, d // 4))
    tail_vals = np.maximum(eigvals[:n_tail], ridge)
    tail_vecs = eigvecs[:, :n_tail]

    train_proj = residuals @ tail_vecs
    train_energy = np.sum((train_proj * train_proj) / tail_vals, axis=1)
    train_energy = np.log1p(np.maximum(train_energy, 0.0))

    global_med = np.median(train_energy)
    global_q75 = np.quantile(train_energy, 0.75)
    global_q25 = np.quantile(train_energy, 0.25)
    global_scale = max(global_q75 - global_q25, 1e-6)

    med = np.full(k, global_med, dtype=np.float64)
    scale = np.full(k, global_scale, dtype=np.float64)
    for c in range(k):
        vals = train_energy[y == c]
        if vals.size >= 4:
            med[c] = np.median(vals)
            scale[c] = max(np.quantile(vals, 0.75) - np.quantile(vals, 0.25), 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    eval_residuals = z - means[pred]
    eval_proj = eval_residuals @ tail_vecs
    eval_energy = np.sum((eval_proj * eval_proj) / tail_vals, axis=1)
    eval_energy = np.log1p(np.maximum(eval_energy, 0.0))

    scores = (eval_energy - med[pred]) / scale[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
