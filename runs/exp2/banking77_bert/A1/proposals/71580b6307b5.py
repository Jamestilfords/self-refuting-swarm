import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The span of class means contains directions most useful for separating known intents, "
    "so in-scope examples should have class-typical residual energy after that span is removed."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0071"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.pooled, dtype=float)
    xev = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = xtr.shape[1]

    global_mean = np.mean(xtr, axis=0)
    counts = np.bincount(labels, minlength=k).astype(float)
    means = np.zeros((k, d), dtype=float)
    np.add.at(means, labels, xtr)
    valid = counts > 0
    means[valid] /= counts[valid, None]
    means[~valid] = global_mean

    weighted_centered_means = (means - global_mean) * np.sqrt(
        np.maximum(counts, 1.0)
    )[:, None]
    _, singular_values, vt = np.linalg.svd(weighted_centered_means, full_matrices=False)
    if singular_values.size:
        threshold = singular_values[0] * 1e-8
        rank = int(np.sum(singular_values > threshold))
    else:
        rank = 0
    basis = vt[:rank]

    train_delta = xtr - means[labels]
    if rank:
        train_residual = train_delta - (train_delta @ basis.T) @ basis
    else:
        train_residual = train_delta
    train_residual_sq = np.sum(train_residual * train_residual, axis=1)

    class_scale = np.zeros(k, dtype=float)
    global_scale = float(np.median(train_residual_sq))
    global_scale = max(global_scale, 1e-12)
    for c in range(k):
        values = train_residual_sq[labels == c]
        class_scale[c] = (
            max(float(np.median(values)), 1e-12) if values.size else global_scale
        )

    predicted = np.argmax(ev.logits, axis=1)
    eval_delta = xev - means[predicted]
    if rank:
        eval_residual = eval_delta - (eval_delta @ basis.T) @ basis
    else:
        eval_residual = eval_delta
    eval_residual_sq = np.sum(eval_residual * eval_residual, axis=1)

    return np.log1p(eval_residual_sq / class_scale[predicted])
