import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The affine subspace spanned by intent prototypes contains directions used to distinguish known intents. "
    "Residual energy outside that supervised subspace, calibrated within the classifier-predicted intent, flags inputs that are not well explained by discriminative in-scope structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0104", "m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    n_classes = int(np.max(y)) + 1

    global_mean = x_train.mean(axis=0)
    means = np.empty((n_classes, x_train.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        members = x_train[y == c]
        means[c] = members.mean(axis=0) if len(members) else global_mean

    centered_means = means - global_mean
    _, singular_values, vt = np.linalg.svd(centered_means, full_matrices=False)
    tol = max(centered_means.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(singular_values > singular_values[0] * tol)) if len(singular_values) else 0
    basis = vt[:max(rank, 1)]

    def residual_energy(x):
        z = x - global_mean
        projection = (z @ basis.T) @ basis
        residual = z - projection
        return np.sum(residual * residual, axis=1)

    train_residual = residual_energy(x_train)
    eval_residual = residual_energy(x_eval)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(x_eval), dtype=np.float64)

    for c in range(n_classes):
        reference = np.sort(train_residual[train_pred == c])
        if len(reference) < 8:
            reference = np.sort(train_residual)
        scores[eval_pred == c] = np.searchsorted(
            reference, eval_residual[eval_pred == c], side="right"
        ) / float(len(reference) + 1)

    return scores
