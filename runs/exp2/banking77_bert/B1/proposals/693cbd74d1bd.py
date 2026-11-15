import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that normal intent variation rarely violates. "
    "Calibrating trailing-subspace residuals by the classifier-assigned intent avoids penalizing naturally broader classes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0134"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_tail = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_tail = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    finite_train = train_tail[np.isfinite(train_tail)]
    fallback_mean = float(np.mean(finite_train)) if finite_train.size else 0.0
    fallback_std = float(np.std(finite_train)) if finite_train.size else 1.0
    fallback_std = max(fallback_std, 1e-6)

    train_tail = np.nan_to_num(
        train_tail, nan=fallback_mean,
        posinf=fallback_mean + 20.0 * fallback_std,
        neginf=fallback_mean - 20.0 * fallback_std,
    )
    eval_tail = np.nan_to_num(
        eval_tail, nan=fallback_mean,
        posinf=fallback_mean + 20.0 * fallback_std,
        neginf=fallback_mean - 20.0 * fallback_std,
    )

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    counts = np.bincount(train_pred, minlength=n_classes).astype(float)
    sums = np.bincount(train_pred, weights=train_tail, minlength=n_classes)
    means = np.full(n_classes, fallback_mean, dtype=float)
    valid = counts >= 5
    means[valid] = sums[valid] / counts[valid]

    centered_sq = (train_tail - means[train_pred]) ** 2
    var_sums = np.bincount(
        train_pred, weights=centered_sq, minlength=n_classes
    )
    scales = np.full(n_classes, fallback_std, dtype=float)
    scales[valid] = np.sqrt(var_sums[valid] / counts[valid])
    scales = np.maximum(scales, 1e-6)

    tail_z = (eval_tail - means[eval_pred]) / scales[eval_pred]

    train_norm = np.linalg.norm(tr.pooled, axis=1)
    eval_norm = np.linalg.norm(ev.pooled, axis=1)
    norm_mean = float(np.mean(train_norm))
    norm_std = max(float(np.std(train_norm)), 1e-6)

    return np.nan_to_num(
        tail_z + 0.02 * (eval_norm - norm_mean) / norm_std,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
