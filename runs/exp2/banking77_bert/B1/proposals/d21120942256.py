import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions encode the representation components that discriminate known intents. "
    "A query unsupported by training examples after removing those directions has atypical within-intent residual structure and should score as OOS."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0053", "m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    mean = x_train.mean(axis=0)
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(classes)).astype(np.float64)

    class_means = np.zeros((len(classes), x_train.shape[1]), dtype=np.float64)
    np.add.at(class_means, inverse, x_train)
    class_means /= np.maximum(counts[:, None], 1.0)

    between = class_means - mean
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    tol = np.finfo(np.float64).eps * max(between.shape) * (
        singular[0] if singular.size else 1.0
    )
    rank = int(np.sum(singular > tol))
    basis = vt[:rank]

    train_residual = x_train - mean
    eval_residual = x_eval - mean
    if rank:
        train_residual -= (train_residual @ basis.T) @ basis
        eval_residual -= (eval_residual @ basis.T) @ basis

    k = min(10, len(train_residual))
    scores = ctx.h.knn(train_residual, eval_residual, k=k, l2=False)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
