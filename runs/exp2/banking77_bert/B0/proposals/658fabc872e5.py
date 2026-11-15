import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions encode variation useful for separating known intents, so projecting them out "
    "isolates the residual structure that should remain typical within a predicted intent. "
    "Large class-conditioned residual deviation flags inputs unsupported by that intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0104"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)
    n_classes = tr.logits.shape[1]

    global_mean = x_train.mean(axis=0)
    class_means = np.empty((n_classes, x_train.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        members = x_train[labels == c]
        class_means[c] = members.mean(axis=0) if len(members) else global_mean

    mean_offsets = class_means - global_mean
    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    rank = int(np.sum(singular_values > singular_values[0] * 1e-10)) if len(singular_values) else 0
    basis = vt[:rank]

    def residual(x):
        centered = x - global_mean
        if rank == 0:
            return centered
        return centered - (centered @ basis.T) @ basis

    train_residual = residual(x_train)
    eval_residual = residual(x_eval)

    residual_centers = np.zeros((n_classes, x_train.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        members = train_residual[labels == c]
        if len(members):
            residual_centers[c] = members.mean(axis=0)

    centered_train_residual = train_residual - residual_centers[labels]
    scale = np.mean(centered_train_residual * centered_train_residual, axis=0)
    floor = max(float(np.mean(scale)) * 1e-8, 1e-12)
    scale = np.maximum(scale, floor)

    predicted = np.argmax(ev.logits, axis=1)
    delta = eval_residual - residual_centers[predicted]
    return np.mean((delta * delta) / scale, axis=1)
