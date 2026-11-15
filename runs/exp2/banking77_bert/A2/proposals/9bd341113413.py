import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that in-scope bottleneck features satisfy tightly. "
    "A predicted-class, shrinkage-stabilized trailing-subspace distance detects violations of those constraints."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007", "m_0021"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]
    tail_dim = min(48, max(8, d // 5))

    counts = np.bincount(y, minlength=k).astype(np.float64)
    global_mean = x.mean(axis=0)
    means = np.empty((k, d), dtype=np.float64)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)
        else:
            means[c] = global_mean

    residual = x - means[y]
    cov = residual.T @ residual / max(len(x) - k, 1)
    cov.flat[::d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1e-12)
    vals, vecs = np.linalg.eigh(cov)
    basis = vecs[:, :tail_dim]

    train_proj = residual @ basis
    global_var = np.mean(train_proj * train_proj, axis=0) + 1e-8
    variances = np.empty((k, tail_dim), dtype=np.float64)
    centers = np.empty(k, dtype=np.float64)
    scales = np.empty(k, dtype=np.float64)

    for c in range(k):
        rc = train_proj[y == c]
        n_c = rc.shape[0]
        if n_c >= 2:
            raw_var = np.mean(rc * rc, axis=0)
            shrink = n_c / (n_c + 20.0)
            variances[c] = shrink * raw_var + (1.0 - shrink) * global_var
            dist = np.sum((rc * rc) / variances[c], axis=1)
        else:
            variances[c] = global_var
            dist = np.sum((train_proj * train_proj) / global_var, axis=1)

        centers[c] = np.median(dist)
        scales[c] = np.median(np.abs(dist - centers[c])) * 1.4826 + 1e-6

    pred = np.argmax(ev.logits, axis=1)
    proj = (z - means[pred]) @ basis
    raw = np.sum((proj * proj) / variances[pred], axis=1)
    return (raw - centers[pred]) / scales[pred]
