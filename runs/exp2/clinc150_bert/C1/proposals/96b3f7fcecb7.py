import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query assigned to an intent should resemble that intent specifically, not merely some class under a marginal density. Per-class shrunk diagonal variances and classwise distance calibration expose examples that are geometrically atypical for their predicted intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n, d = x.shape
    k = tr.logits.shape[1]
    counts = np.bincount(y, minlength=k).astype(np.float64)

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)

    for c in range(k):
        xc = x[y == c]
        nc = xc.shape[0]
        if nc == 0:
            means[c] = global_mean
            variances[c] = global_var
            continue

        mu = np.mean(xc, axis=0)
        means[c] = mu
        if nc > 1:
            local_var = np.mean((xc - mu) ** 2, axis=0)
        else:
            local_var = global_var

        shrink = nc / (nc + 24.0)
        variances[c] = shrink * local_var + (1.0 - shrink) * global_var

    variances = np.maximum(variances, global_var[None, :] * 1e-5)
    inv_var = 1.0 / variances

    train_centered = x - means[y]
    train_dist = np.sum(train_centered * train_centered * inv_var[y], axis=1)

    dist_mean = np.zeros(k, dtype=np.float64)
    dist_std = np.ones(k, dtype=np.float64)
    for c in range(k):
        dc = train_dist[y == c]
        if dc.size:
            dist_mean[c] = np.mean(dc)
            dist_std[c] = max(np.std(dc), 1e-6)

    pred = np.argmax(ev.logits, axis=1).astype(np.int64)
    centered = z - means[pred]
    raw_dist = np.sum(centered * centered * inv_var[pred], axis=1)

    calibrated = (raw_dist - dist_mean[pred]) / dist_std[pred]
    confidence_penalty = -np.max(ev.logits, axis=1)
    return calibrated + 0.02 * confidence_penalty
