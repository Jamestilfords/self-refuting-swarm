import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture the representation components most useful for separating known intents. "
    "After removing those directions, OOS queries should have poorer local support in the remaining nuisance and residual feature geometry."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0045", "m_0049"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    global_mean = np.mean(x, axis=0)
    classes = np.unique(y)
    means = np.empty((len(classes), x.shape[1]), dtype=float)
    weights = np.empty(len(classes), dtype=float)

    for i, c in enumerate(classes):
        members = x[y == c]
        means[i] = np.mean(members, axis=0)
        weights[i] = len(members)

    centered_means = means - global_mean
    scale = np.std(x - global_mean, axis=0)
    scale = np.maximum(scale, 1e-6)
    weighted_means = centered_means * np.sqrt(weights[:, None]) / scale

    _, _, vt = np.linalg.svd(weighted_means, full_matrices=False)
    rank = min(vt.shape[0], 96)
    basis = vt[:rank] / scale[None, :]

    xc = x - global_mean
    zc = z - global_mean

    train_disc = (xc / scale) @ vt[:rank].T
    eval_disc = (zc / scale) @ vt[:rank].T
    train_scaled = xc / scale
    eval_scaled = zc / scale

    train_residual = train_scaled - train_disc @ vt[:rank]
    eval_residual = eval_scaled - eval_disc @ vt[:rank]

    train_residual = np.nan_to_num(train_residual, nan=0.0, posinf=1e6, neginf=-1e6)
    eval_residual = np.nan_to_num(eval_residual, nan=0.0, posinf=1e6, neginf=-1e6)

    return ctx.h.knn(train_residual, eval_residual, k=15, l2=True)
