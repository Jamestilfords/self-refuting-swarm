import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-intent directions encode constraints that ordinary in-scope examples satisfy. This uses classifier-predicted-class prototypes and train-calibrated trailing-subspace residuals, so unusually large residual energy is scored as OOS."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0091"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(x, axis=0)
    means = np.empty((k, d), dtype=float)
    centered = np.empty_like(x)

    for c in range(k):
        mask = y == c
        if np.any(mask):
            means[c] = np.mean(x[mask], axis=0)
        else:
            means[c] = global_mean
    centered[:] = x - means[y]

    cov = (centered.T @ centered) / max(n - k, 1)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    q = min(32, max(1, d // 8))
    trailing = eigvecs[:, :q]
    floor = max(float(np.mean(np.maximum(eigvals[:q], 0.0))) * 1e-3, 1e-8)
    scales = np.sqrt(np.maximum(eigvals[:q], floor))

    tr_resid = ((x - means[pred_tr]) @ trailing) / scales
    ev_resid = ((z - means[pred_ev]) @ trailing) / scales
    tr_energy = np.mean(tr_resid * tr_resid, axis=1)
    ev_energy = np.mean(ev_resid * ev_resid, axis=1)

    global_med = float(np.median(tr_energy))
    global_hi = float(np.quantile(tr_energy, 0.90))
    global_scale = max(global_hi - global_med, 1e-6)

    med = np.full(k, global_med, dtype=float)
    scale = np.full(k, global_scale, dtype=float)
    for c in range(k):
        vals = tr_energy[pred_tr == c]
        if vals.size >= 8:
            m = float(np.median(vals))
            h = float(np.quantile(vals, 0.90))
            med[c] = m
            scale[c] = max(h - m, 1e-6)

    scores = (ev_energy - med[pred_ev]) / scale[pred_ev]
    return np.clip(scores, -1e6, 1e6)
