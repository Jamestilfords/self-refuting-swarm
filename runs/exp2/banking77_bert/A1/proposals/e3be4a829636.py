import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions encode variations that are tightly constrained across known intents. A robust log-penalty for predicted-class residuals in those directions emphasizes atypical constraint violations without reducing to the usual quadratic trailing-energy ranking."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0073", "m_0077"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.empty((k, d), dtype=float)
    for c in range(k):
        xc = x[y == c]
        means[c] = np.mean(xc, axis=0) if xc.shape[0] else global_mean

    train_resid = x - means[y]
    cov = (train_resid.T @ train_resid) / max(train_resid.shape[0] - 1, 1)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)

    n_tail = min(64, max(8, d // 4))
    tail_vals = np.maximum(vals[:n_tail], 1e-8)
    tail_vecs = vecs[:, :n_tail]

    pred = np.argmax(ev.logits, axis=1)
    eval_resid = z - means[pred]
    proj = eval_resid @ tail_vecs
    standardized_sq = (proj * proj) / tail_vals[None, :]

    # Log penalties prevent a single numerically extreme direction from
    # dominating while preserving sensitivity to many violated constraints.
    out = np.sum(np.log1p(standardized_sq), axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
