import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode feature changes that in-scope examples rarely make. A predicted-class-calibrated maximum standardized tail-coordinate deviation highlights a single strongly violated constrained direction rather than reproducing a summed Mahalanobis score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]
    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)

    for c in range(k):
        mask = y == c
        if np.any(mask):
            means[c] = x[mask].mean(axis=0)
        else:
            means[c] = x.mean(axis=0)

    residuals = x - means[y]
    denom = max(n - np.count_nonzero(counts), 1)
    cov = (residuals.T @ residuals) / denom
    scale = max(float(np.trace(cov)) / max(d, 1), 1e-10)
    cov.flat[::d + 1] += scale * 1e-6

    vals, vecs = np.linalg.eigh(cov)
    tail_dim = min(80, d)
    floor = scale * 1e-4
    tail_vals = np.maximum(vals[:tail_dim], floor)
    tail_vecs = vecs[:, :tail_dim]

    train_proj = (residuals @ tail_vecs) / np.sqrt(tail_vals)
    train_stat = np.max(np.abs(train_proj), axis=1)

    med = np.zeros(k, dtype=float)
    spread = np.ones(k, dtype=float)
    global_med = float(np.median(train_stat))
    global_iqr = float(np.percentile(train_stat, 75) - np.percentile(train_stat, 25))
    global_iqr = max(global_iqr, 1e-6)

    for c in range(k):
        vals_c = train_stat[y == c]
        if vals_c.size >= 4:
            med[c] = np.median(vals_c)
            iqr = np.percentile(vals_c, 75) - np.percentile(vals_c, 25)
            spread[c] = max(float(iqr), global_iqr * 0.1, 1e-6)
        else:
            med[c] = global_med
            spread[c] = global_iqr

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    ev_residual = z - means[pred]
    ev_proj = (ev_residual @ tail_vecs) / np.sqrt(tail_vals)
    ev_stat = np.max(np.abs(ev_proj), axis=1)

    scores = (ev_stat - med[pred]) / spread[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
