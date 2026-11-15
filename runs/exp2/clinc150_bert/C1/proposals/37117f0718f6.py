import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions describe the discriminative subspace used to separate in-scope intents. Queries with unusually large representation energy orthogonal to that subspace are likely to contain factors not explained by known intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    classes, inverse = np.unique(labels, return_inverse=True)
    means = np.zeros((classes.size, x.shape[1]), dtype=float)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    np.add.at(means, inverse, x)
    means /= counts[:, None]

    center = np.average(means, axis=0, weights=counts)
    between = (means - center) * np.sqrt(counts)[:, None]
    _, singular, vt = np.linalg.svd(between, full_matrices=False)

    power = singular * singular
    total = float(power.sum())
    if total > 1e-12:
        rank = int(np.searchsorted(np.cumsum(power) / total, 0.98) + 1)
        rank = max(1, min(rank, vt.shape[0]))
        basis = vt[:rank]
        tr_delta = x - center
        ev_delta = z - center
        tr_resid = tr_delta - (tr_delta @ basis.T) @ basis
        ev_resid = ev_delta - (ev_delta @ basis.T) @ basis
    else:
        tr_resid = x - center
        ev_resid = z - center

    train_score = np.sum(tr_resid * tr_resid, axis=1)
    location = float(np.median(train_score))
    scale = float(np.median(np.abs(train_score - location)))
    scale = max(scale, 1e-8)
    return np.sum(ev_resid * ev_resid, axis=1) / scale
