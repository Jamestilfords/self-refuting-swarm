import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Each intent has a characteristic diagonal residual-variance profile, including both total "
    "standardized radius and how concentrated that radius is across coordinates. Queries whose "
    "classifier-selected class exhibits an atypical residual profile are scored as out-of-scope."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0094", "m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    k = tr.logits.shape[1]
    d = x.shape[1]
    eps = 1e-8

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + eps

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    counts = np.bincount(y, minlength=k).astype(np.float64)

    for c in range(k):
        xc = x[y == c]
        if xc.shape[0] == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            local_var = np.var(xc, axis=0)
            shrink = xc.shape[0] / (xc.shape[0] + 20.0)
            variances[c] = shrink * local_var + (1.0 - shrink) * global_var + eps

    train_radius = np.empty(x.shape[0], dtype=np.float64)
    train_concentration = np.empty(x.shape[0], dtype=np.float64)
    for c in range(k):
        mask = y == c
        if not np.any(mask):
            continue
        q = (x[mask] - means[c]) ** 2 / variances[c]
        total = np.sum(q, axis=1) + eps
        train_radius[mask] = np.log(total / d)
        train_concentration[mask] = np.log(np.sum(q * q, axis=1) / (total * total) * d + eps)

    stat_mean = np.zeros((k, 2), dtype=np.float64)
    stat_var = np.ones((k, 2), dtype=np.float64)
    for c in range(k):
        mask = y == c
        if np.any(mask):
            vals = np.column_stack((train_radius[mask], train_concentration[mask]))
            stat_mean[c] = np.mean(vals, axis=0)
            stat_var[c] = np.var(vals, axis=0) + 1e-4

    qev = (z - means[pred_ev]) ** 2 / variances[pred_ev]
    total_ev = np.sum(qev, axis=1) + eps
    radius_ev = np.log(total_ev / d)
    concentration_ev = np.log(np.sum(qev * qev, axis=1) / (total_ev * total_ev) * d + eps)

    mu = stat_mean[pred_ev]
    var = stat_var[pred_ev]
    score_out = ((radius_ev - mu[:, 0]) ** 2 / var[:, 0] +
                 (concentration_ev - mu[:, 1]) ** 2 / var[:, 1])
    return np.nan_to_num(score_out, nan=0.0, posinf=1e12, neginf=0.0)
