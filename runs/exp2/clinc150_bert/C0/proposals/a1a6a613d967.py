import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class means define directions most associated with intent identity; OOS inputs should retain "
    "unusually large energy in the orthogonal residual subspace after those directions are removed."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    global_mean = x_train.mean(axis=0)
    class_means = np.empty((n_classes, x_train.shape[1]), dtype=float)
    for c in range(n_classes):
        members = x_train[labels == c]
        class_means[c] = members.mean(axis=0) if len(members) else global_mean

    mean_offsets = class_means - global_mean
    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    if singular_values.size:
        threshold = max(singular_values[0] * 1e-8, 1e-12)
        rank = int(np.sum(singular_values > threshold))
    else:
        rank = 0
    basis = vt[:rank]

    train_centered = x_train - global_mean
    eval_centered = x_eval - global_mean
    if rank:
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_residual = eval_centered - (eval_centered @ basis.T) @ basis
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    train_energy = np.sum(train_residual * train_residual, axis=1)
    eval_energy = np.sum(eval_residual * eval_residual, axis=1)

    class_scale = np.empty(n_classes, dtype=float)
    for c in range(n_classes):
        values = train_energy[labels == c]
        if len(values):
            class_scale[c] = np.median(values)
        else:
            class_scale[c] = np.median(train_energy)

    fallback = max(float(np.median(train_energy)), 1e-12)
    class_scale = np.maximum(class_scale, fallback * 1e-6 + 1e-12)
    predicted = np.argmax(ev.logits, axis=1)
    residual_score = eval_energy / class_scale[predicted]

    train_norm = np.sum(train_centered * train_centered, axis=1)
    norm_scale = max(float(np.median(train_norm)), 1e-12)
    eval_norm = np.sum(eval_centered * eval_centered, axis=1)
    return np.asarray(residual_score + 1e-6 * eval_norm / norm_scale, dtype=float)
