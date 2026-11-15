import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture the pooled-representation variation that separates known intents. "
    "Large energy in their orthogonal complement indicates encoder content unsupported by the intent-discriminative subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    mean = x_train.mean(axis=0)
    n_classes = int(max(labels.max() + 1, tr.logits.shape[1]))

    class_means = np.empty((n_classes, x_train.shape[1]), dtype=float)
    for c in range(n_classes):
        members = x_train[labels == c]
        class_means[c] = members.mean(axis=0) if len(members) else mean

    between = class_means - mean
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    tol = max(between.shape) * np.finfo(float).eps * (
        singular[0] if singular.size else 1.0
    )
    rank = int(np.sum(singular > tol))

    if rank == 0:
        train_residual = np.sum((x_train - mean) ** 2, axis=1)
        eval_residual = np.sum((x_eval - mean) ** 2, axis=1)
    else:
        basis = vt[:rank]
        train_centered = x_train - mean
        eval_centered = x_eval - mean
        train_proj_sq = np.sum((train_centered @ basis.T) ** 2, axis=1)
        eval_proj_sq = np.sum((eval_centered @ basis.T) ** 2, axis=1)
        train_residual = np.maximum(
            np.sum(train_centered ** 2, axis=1) - train_proj_sq, 0.0
        )
        eval_residual = np.maximum(
            np.sum(eval_centered ** 2, axis=1) - eval_proj_sq, 0.0
        )

    pred = np.argmax(ev.logits, axis=1)
    global_scale = np.median(train_residual) + 1e-12
    scores = np.empty(len(x_eval), dtype=float)

    for c in range(n_classes):
        mask_eval = pred == c
        if not np.any(mask_eval):
            continue
        reference = np.sort(train_residual[labels == c])
        if reference.size < 2:
            reference = np.sort(train_residual)
        tail_rank = np.searchsorted(reference, eval_residual[mask_eval], side="right")
        empirical_tail = tail_rank / float(reference.size)
        continuous = eval_residual[mask_eval] / (
            eval_residual[mask_eval] + global_scale
        )
        scores[mask_eval] = empirical_tail + 1e-6 * continuous

    return np.nan_to_num(scores, nan=0.0, posinf=1.000001, neginf=0.0)
