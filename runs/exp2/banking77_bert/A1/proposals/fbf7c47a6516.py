import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "After removing the classifier-predicted intent prototype, in-scope queries should "
    "lie near the principal within-intent variation subspace, while OOS queries retain larger residual norm."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051", "m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    n_classes = int(max(np.max(y), tr.logits.shape[1] - 1)) + 1
    global_mean = np.mean(x, axis=0)
    means = np.empty((n_classes, x.shape[1]), dtype=np.float64)

    for c in range(n_classes):
        mask = y == c
        means[c] = np.mean(x[mask], axis=0) if np.any(mask) else global_mean

    centered_train = x - means[y]
    _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    rank = min(96, vt.shape[0], vt.shape[1])
    basis = vt[:rank]

    train_residual = centered_train - (centered_train @ basis.T) @ basis
    train_norm = np.sqrt(np.maximum(np.sum(train_residual * train_residual, axis=1), 0.0))

    pred_train = np.argmax(tr.logits, axis=1)
    medians = np.empty(n_classes, dtype=np.float64)
    scales = np.empty(n_classes, dtype=np.float64)
    default_median = np.median(train_norm)
    default_scale = np.median(np.abs(train_norm - default_median)) + 1e-6

    for c in range(n_classes):
        vals = train_norm[pred_train == c]
        if vals.size >= 5:
            medians[c] = np.median(vals)
            scales[c] = np.median(np.abs(vals - medians[c])) + 1e-6
        else:
            medians[c] = default_median
            scales[c] = default_scale

    pred_eval = np.argmax(ev.logits, axis=1)
    pred_eval = np.clip(pred_eval, 0, n_classes - 1)
    centered_eval = z - means[pred_eval]
    eval_residual = centered_eval - (centered_eval @ basis.T) @ basis
    eval_norm = np.sqrt(np.maximum(np.sum(eval_residual * eval_residual, axis=1), 0.0))

    scores = (eval_norm - medians[pred_eval]) / scales[pred_eval]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
