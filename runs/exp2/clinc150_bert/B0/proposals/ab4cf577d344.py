import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean subspace suppresses directions used mainly to discriminate "
    "known intents, so cosine neighborhood distance in the residual space measures support "
    "for variation not explained by in-scope intent identity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0028", "m_0032"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    classes = np.unique(tr.labels)
    means = np.vstack([tr.pooled[tr.labels == c].mean(axis=0) for c in classes])
    centered_means = means - means.mean(axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(centered_means, full_matrices=False)
    variance = singular_values ** 2
    total = variance.sum()

    if total > 1e-12:
        cumulative = np.cumsum(variance) / total
        n_components = int(np.searchsorted(cumulative, 0.90) + 1)
        n_components = max(1, min(n_components, vt.shape[0]))
        basis = vt[:n_components].T
        train_residual = tr.pooled - tr.pooled @ basis @ basis.T
        eval_residual = ev.pooled - ev.pooled @ basis @ basis.T
    else:
        train_residual = tr.pooled - tr.pooled.mean(axis=0, keepdims=True)
        eval_residual = ev.pooled - tr.pooled.mean(axis=0, keepdims=True)

    return ctx.h.knn(train_residual, eval_residual, k=15, l2=True)
