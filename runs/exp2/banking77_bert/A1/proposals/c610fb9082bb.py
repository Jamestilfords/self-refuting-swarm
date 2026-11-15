import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should fit a query better than its strongest logit rival under "
    "that intent's own shrunk diagonal covariance. OOS queries can retain a confident prediction "
    "while the runner-up intent provides equal or better heteroscedastic feature support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0099", "m_0111", "m_0121"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    sq_sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sq_sums, y, centered * centered)
    class_var = sq_sums / np.maximum(counts[:, None] - 1.0, 1.0)

    global_var = np.var(x, axis=0, ddof=1)
    shrink = counts[:, None] / (counts[:, None] + 20.0)
    var = shrink * class_var + (1.0 - shrink) * global_var[None, :]
    floor = np.maximum(np.median(global_var) * 1e-4, 1e-8)
    var = np.maximum(var, floor)

    logits = np.asarray(ev.logits)
    pred = np.argmax(logits, axis=1)
    masked = logits.copy()
    masked[np.arange(masked.shape[0]), pred] = -np.inf
    rival = np.argmax(masked, axis=1)

    pred_resid = z - means[pred]
    rival_resid = z - means[rival]
    pred_dist = np.sum(pred_resid * pred_resid / var[pred], axis=1) / d
    rival_dist = np.sum(rival_resid * rival_resid / var[rival], axis=1) / d

    return pred_dist - rival_dist
