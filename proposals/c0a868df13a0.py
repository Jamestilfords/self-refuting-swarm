import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The span of class means captures directions used to distinguish known intents. "
    "After removing that discriminative span, unusually large residual content for the classifier-selected intent indicates unsupported semantic variation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0227"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    center = x_train.mean(axis=0)
    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    means = np.zeros((n_classes, dim), dtype=float)
    np.add.at(means, labels, x_train)
    valid = counts > 0
    means[valid] /= counts[valid, None]
    means[~valid] = center

    class_offsets = means - center
    _, singular, vt = np.linalg.svd(class_offsets, full_matrices=False)
    rank = int(np.sum(singular > (singular[0] * 1e-8))) if singular.size else 0
    use_dim = min(rank, 64)

    train_centered = x_train - center
    eval_centered = x_eval - center

    if use_dim > 0:
        basis = vt[:use_dim].T
        train_proj_sq = np.sum((train_centered @ basis) ** 2, axis=1)
        eval_proj_sq = np.sum((eval_centered @ basis) ** 2, axis=1)
    else:
        train_proj_sq = np.zeros(x_train.shape[0], dtype=float)
        eval_proj_sq = np.zeros(x_eval.shape[0], dtype=float)

    train_total_sq = np.sum(train_centered * train_centered, axis=1)
    eval_total_sq = np.sum(eval_centered * eval_centered, axis=1)
    train_residual = np.sqrt(np.maximum(train_total_sq - train_proj_sq, 0.0))
    eval_residual = np.sqrt(np.maximum(eval_total_sq - eval_proj_sq, 0.0))

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    global_sorted = np.sort(train_residual)
    global_scale = np.median(train_residual) + 1e-8

    out = np.empty(eval_residual.shape[0], dtype=float)
    for c in range(n_classes):
        mask = eval_pred == c
        if not np.any(mask):
            continue
        reference = train_residual[train_pred == c]
        if reference.size < 8:
            reference = global_sorted
        else:
            reference = np.sort(reference)
        rank_above = reference.size - np.searchsorted(
            reference, eval_residual[mask], side="right"
        )
        tail = (rank_above + 1.0) / (reference.size + 2.0)
        out[mask] = -np.log(tail) + 1e-6 * (eval_residual[mask] / global_scale)

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
