import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A ViM-style residual norm measures encoder content outside the global in-scope principal subspace. "
    "Calibrating that residual by the classifier-selected intent accounts for ordinary intent-specific variation while retaining unsupported residual content."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0197", "m_0200"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    n, d = x_train.shape

    mean = x_train.mean(axis=0)
    centered_train = x_train - mean

    # Deterministic PCA through the training covariance.  Keeping a substantial
    # leading subspace leaves a residual space analogous to ViM's null space.
    cov = (centered_train.T @ centered_train) / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n_components = min(256, d, max(n - 1, 1))
    basis = eigvecs[:, -n_components:]

    def residual_norm(x):
        centered = x - mean
        total_sq = np.sum(centered * centered, axis=1)
        projected = centered @ basis
        explained_sq = np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(total_sq - explained_sq, 0.0))

    train_residual = residual_norm(x_train)
    eval_residual = residual_norm(x_eval)

    # Calibration follows the class selected by the classifier rather than the
    # ground-truth label, matching the class-selection mechanism at evaluation.
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_med = np.median(train_residual)
    global_mad = np.median(np.abs(train_residual - global_med))
    floor = max(1e-8, (np.mean(np.abs(train_residual)) + 1.0) * 1e-6)
    global_scale = max(1.4826 * global_mad, floor)

    medians = np.full(k, global_med, dtype=np.float64)
    scales = np.full(k, global_scale, dtype=np.float64)

    for cls in range(k):
        values = train_residual[train_pred == cls]
        if values.size >= 8:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[cls] = med
            scales[cls] = max(1.4826 * mad, floor)

    scores = (eval_residual - medians[eval_pred]) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
