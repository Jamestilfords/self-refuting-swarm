import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "The class-mean subspace captures directions used to distinguish known intents, but in-scope residual energy is also conditioned on where a query lies within that discriminative subspace. A train-fitted ridge model flags excess residual energy beyond that expected from the query's discriminative coordinates."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n, d = x.shape
    labels, inverse = np.unique(y, return_inverse=True)
    c = len(labels)
    counts = np.bincount(inverse, minlength=c).astype(np.float64)

    mean = np.mean(x, axis=0)
    xc = x - mean
    zc = z - mean

    class_means = np.zeros((c, d), dtype=np.float64)
    np.add.at(class_means, inverse, x)
    class_means /= np.maximum(counts[:, None], 1.0)

    mean_dirs = class_means - mean
    weighted_dirs = mean_dirs * np.sqrt(np.maximum(counts, 1.0))[:, None]
    _, singular, vt = np.linalg.svd(weighted_dirs, full_matrices=False)

    if singular.size:
        threshold = max(float(singular[0]) * 1e-8, 1e-10)
        rank = int(np.sum(singular > threshold))
    else:
        rank = 0
    rank = max(1, min(rank, vt.shape[0]))
    basis = vt[:rank].T

    train_disc = xc @ basis
    eval_disc = zc @ basis
    train_resid = xc - train_disc @ basis.T
    eval_resid = zc - eval_disc @ basis.T

    target = np.log1p(np.sum(train_resid * train_resid, axis=1))
    scale = np.sqrt(np.mean(train_disc * train_disc, axis=0))
    scale = np.maximum(scale, 1e-6)
    train_features = train_disc / scale
    eval_features = eval_disc / scale

    train_features = np.concatenate(
        [np.ones((n, 1)), train_features], axis=1
    )
    eval_features = np.concatenate(
        [np.ones((len(z), 1)), eval_features], axis=1
    )

    gram = train_features.T @ train_features
    ridge = 1e-2 * np.trace(gram) / max(gram.shape[0], 1)
    gram.flat[:: gram.shape[0] + 1] += max(ridge, 1e-6)
    coef = np.linalg.solve(gram, train_features.T @ target)

    fitted = train_features @ coef
    residual = target - fitted
    resid_scale = np.sqrt(np.mean(residual * residual))
    resid_scale = max(float(resid_scale), 1e-6)

    observed = np.log1p(np.sum(eval_resid * eval_resid, axis=1))
    expected = eval_features @ coef
    scores = (observed - expected) / resid_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
