import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean discriminative subspace leaves variation that is weakly explained by known intent separation. "
    "OOS queries should have poorly supported local neighborhoods in this residual representation even when their discriminative projection resembles an in-scope intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0023", "m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.pooled, axis=0, keepdims=True)
    classes = np.unique(tr.labels)
    class_means = np.vstack([
        np.mean(tr.pooled[tr.labels == c], axis=0) for c in classes
    ])
    between = class_means - np.mean(class_means, axis=0, keepdims=True)

    _, s, vt = np.linalg.svd(between, full_matrices=False)
    tol = np.max(s) * max(between.shape) * np.finfo(float).eps
    rank = int(np.sum(s > tol))
    rank = max(1, min(rank, 64))
    basis = vt[:rank].T

    train_centered = tr.pooled - mean
    eval_centered = ev.pooled - mean
    train_residual = train_centered - (train_centered @ basis) @ basis.T
    eval_residual = eval_centered - (eval_centered @ basis) @ basis.T

    neighborhood = ctx.h.knn(train_residual, eval_residual, k=10, l2=False)
    residual_energy = np.sqrt(np.sum(eval_residual * eval_residual, axis=1))
    return neighborhood + 1e-6 * residual_energy
