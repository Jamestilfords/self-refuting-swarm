import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Intent mean differences define a low-dimensional discriminative subspace, while variation orthogonal "
    "to it is less explained by any in-scope intent. OOS queries should therefore have unusually large "
    "class-calibrated residual energy outside this between-intent subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0033"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.empty((k, d), dtype=np.float64)
    counts = np.bincount(y, minlength=k)
    for c in range(k):
        if counts[c] > 0:
            means[c] = np.mean(x[y == c], axis=0)
        else:
            means[c] = global_mean

    between = means - global_mean
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    tol = max(float(singular[0]) if singular.size else 0.0, 1.0) * 1e-10
    rank = int(np.sum(singular > tol))
    if rank == 0:
        basis = np.zeros((d, 0), dtype=np.float64)
    else:
        basis = vt[:rank].T

    train_centered = x - global_mean
    eval_centered = z - global_mean
    if rank:
        train_proj = train_centered @ basis
        eval_proj = eval_centered @ basis
        train_resid_sq = np.maximum(
            np.sum(train_centered * train_centered, axis=1)
            - np.sum(train_proj * train_proj, axis=1),
            0.0,
        )
        eval_resid_sq = np.maximum(
            np.sum(eval_centered * eval_centered, axis=1)
            - np.sum(eval_proj * eval_proj, axis=1),
            0.0,
        )
    else:
        train_resid_sq = np.sum(train_centered * train_centered, axis=1)
        eval_resid_sq = np.sum(eval_centered * eval_centered, axis=1)

    train_resid = np.sqrt(train_resid_sq + 1e-12)
    eval_resid = np.sqrt(eval_resid_sq + 1e-12)

    class_loc = np.empty(k, dtype=np.float64)
    class_scale = np.empty(k, dtype=np.float64)
    for c in range(k):
        vals = train_resid[y == c]
        if vals.size:
            loc = np.median(vals)
            mad = np.median(np.abs(vals - loc))
            class_loc[c] = loc
            class_scale[c] = max(1.4826 * mad, 1e-6)
        else:
            class_loc[c] = np.median(train_resid)
            class_scale[c] = max(
                1.4826 * np.median(np.abs(train_resid - class_loc[c])),
                1e-6,
            )

    pred = np.argmax(ev.logits, axis=1)
    return (eval_resid - class_loc[pred]) / class_scale[pred]
