import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "After removing the class-mean discriminative subspace, in-scope examples should have "
    "residual projections that remain within class-supported directional ranges. Random residual "
    "directions expose off-manifold deviations that may be missed by an aggregate residual norm."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0196", "m_0207"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)
    k = tr.logits.shape[1]

    center = np.mean(x, axis=0)
    means = np.zeros((k, x.shape[1]), dtype=float)
    counts = np.bincount(y, minlength=k)
    for c in range(k):
        if counts[c] > 0:
            means[c] = np.mean(x[y == c], axis=0)
        else:
            means[c] = center

    mean_offsets = means - np.mean(means, axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    rank = int(np.sum(s > (s[0] * 1e-8 if s.size else 0.0)))
    u = vt[:rank].T if rank > 0 else np.zeros((x.shape[1], 0), dtype=float)

    xc = x - center
    zc = z - center
    if rank > 0:
        xr = xc - (xc @ u) @ u.T
        zr = zc - (zc @ u) @ u.T
    else:
        xr = xc
        zr = zc

    rng = np.random.default_rng(271828)
    n_dirs = min(48, x.shape[1] - rank)
    raw = rng.standard_normal((x.shape[1], n_dirs))
    if rank > 0:
        raw = raw - u @ (u.T @ raw)
    directions, _ = np.linalg.qr(raw)
    directions = directions[:, :n_dirs]

    train_proj = xr @ directions
    eval_proj = zr @ directions

    med = np.zeros((k, n_dirs), dtype=float)
    tail = np.zeros((k, n_dirs), dtype=float)
    scale = np.zeros((k, n_dirs), dtype=float)
    global_med = np.median(train_proj, axis=0)
    global_abs = np.abs(train_proj - global_med)
    global_tail = np.quantile(global_abs, 0.95, axis=0)
    global_scale = np.quantile(global_abs, 0.75, axis=0) + 1e-6

    for c in range(k):
        vals = train_proj[y == c]
        if vals.shape[0] >= 4:
            mc = np.median(vals, axis=0)
            dev = np.abs(vals - mc)
            med[c] = mc
            tail[c] = np.quantile(dev, 0.95, axis=0)
            scale[c] = np.quantile(dev, 0.75, axis=0) + 1e-6
        else:
            med[c] = global_med
            tail[c] = global_tail
            scale[c] = global_scale

    pred = np.argmax(ev.logits, axis=1)
    dev = np.abs(eval_proj - med[pred])
    excess = np.maximum(dev - tail[pred], 0.0) / scale[pred]
    out = np.mean(np.log1p(excess), axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)
