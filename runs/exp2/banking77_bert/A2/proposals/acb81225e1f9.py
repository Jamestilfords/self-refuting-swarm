import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "The span of class means contains directions most useful for separating known intents, "
    "so OOS inputs may violate class-conditional constraints in its orthogonal residual space. "
    "Predicted-intent calibration makes residual surprise comparable across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0111", "m_0113"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)
    k = tr.logits.shape[1]

    global_mean = np.mean(x, axis=0, keepdims=True)
    centered = x - global_mean

    class_means = np.repeat(global_mean, k, axis=0)
    present = np.zeros(k, dtype=bool)
    for c in range(k):
        mask = y == c
        if np.any(mask):
            class_means[c] = np.mean(x[mask], axis=0)
            present[c] = True

    mean_offsets = class_means[present] - global_mean
    if mean_offsets.shape[0] > 0:
        _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)
        tol = max(mean_offsets.shape) * np.finfo(float).eps * (
            singular[0] if singular.size else 1.0
        )
        rank = int(np.sum(singular > tol))
        basis = vt[:rank]
    else:
        basis = np.empty((0, x.shape[1]), dtype=float)

    def residual(a):
        ac = a - global_mean
        if basis.shape[0] == 0:
            return ac
        return ac - (ac @ basis.T) @ basis

    rx = residual(x)
    rz = residual(z)

    global_var = np.var(rx, axis=0) + 1e-6
    class_var = np.repeat(global_var[None, :], k, axis=0)
    for c in range(k):
        mask = y == c
        if np.sum(mask) >= 2:
            class_var[c] = np.var(rx[mask], axis=0) + 1e-6

    train_var = class_var[np.clip(y, 0, k - 1)]
    train_raw = np.mean(np.log1p((rx * rx) / train_var), axis=1)

    med = np.zeros(k, dtype=float)
    scale = np.ones(k, dtype=float)
    global_scale = np.percentile(train_raw, 75) - np.percentile(train_raw, 25)
    if not np.isfinite(global_scale) or global_scale < 1e-6:
        global_scale = np.std(train_raw) + 1e-6

    for c in range(k):
        vals = train_raw[y == c]
        if vals.size:
            med[c] = np.median(vals)
            s = np.percentile(vals, 75) - np.percentile(vals, 25)
            scale[c] = s if s > 1e-6 and np.isfinite(s) else global_scale
        else:
            med[c] = np.median(train_raw)
            scale[c] = global_scale

    pred = np.argmax(ev.logits, axis=1)
    eval_var = class_var[pred]
    raw = np.mean(np.log1p((rz * rz) / eval_var), axis=1)
    out = (raw - med[pred]) / scale[pred]
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
