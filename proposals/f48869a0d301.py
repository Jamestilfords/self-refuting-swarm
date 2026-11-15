import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture intent-discriminative variation, so the orthogonal "
    "residual should retain locally supported nuisance and semantic variation. Global angular "
    "kNN support in that residual space flags novel content without rewarding merely large norm."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0207", "m_0014"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.feats, axis=0)
    x_train = tr.feats - mean
    x_eval = ev.feats - mean

    classes = np.unique(tr.labels)
    class_means = np.vstack([
        np.mean(x_train[tr.labels == c], axis=0) for c in classes
    ])
    class_means = class_means - np.mean(class_means, axis=0, keepdims=True)

    _, singular, vt = np.linalg.svd(class_means, full_matrices=False)
    if singular.size == 0 or singular[0] <= 1e-12:
        basis = np.empty((0, x_train.shape[1]), dtype=x_train.dtype)
    else:
        positive = singular > (singular[0] * 1e-8)
        usable = singular[positive]
        cumulative = np.cumsum(usable * usable)
        rank = int(np.searchsorted(cumulative, 0.95 * cumulative[-1]) + 1)
        basis = vt[:rank]

    if basis.shape[0]:
        train_residual = x_train - (x_train @ basis.T) @ basis
        eval_residual = x_eval - (x_eval @ basis.T) @ basis
    else:
        train_residual = x_train
        eval_residual = x_eval

    train_norm = np.sqrt(np.sum(train_residual * train_residual, axis=1, keepdims=True))
    eval_norm = np.sqrt(np.sum(eval_residual * eval_residual, axis=1, keepdims=True))
    train_unit = train_residual / np.maximum(train_norm, 1e-12)
    eval_unit = eval_residual / np.maximum(eval_norm, 1e-12)

    train_radius = ctx.h.knn(train_unit, train_unit, k=11, l2=False)
    eval_radius = ctx.h.knn(train_unit, eval_unit, k=10, l2=False)
    normalizer = ctx.h.normalizer("ecdf", train_radius)
    return normalizer(eval_radius)
