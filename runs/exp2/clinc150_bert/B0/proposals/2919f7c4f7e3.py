import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions represent constrained intent-specific variation, so "
    "large residual energy in those directions is atypical for an item assigned to a predicted intent. "
    "A predicted-class conditional empirical tail probability calibrates this anomaly without using eval-set statistics."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0034", "m_0035", "m_0036"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z_tr = np.asarray(tr.logits, dtype=np.float64)
    x_ev = np.asarray(ev.feats, dtype=np.float64)
    z_ev = np.asarray(ev.logits, dtype=np.float64)

    n, d = x.shape
    k = z_tr.shape[1]
    counts = np.bincount(y, minlength=k).astype(np.float64)
    global_mean = x.mean(axis=0)

    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    valid = counts > 0
    means[valid] /= counts[valid, None]
    means[~valid] = global_mean

    within = x - means[y]
    cov = (within.T @ within) / max(n - k, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    tail_dim = max(16, min(64, d // 4))
    tail_vals = eigvals[:tail_dim]
    tail_vecs = eigvecs[:, :tail_dim]
    positive = eigvals[eigvals > 1e-12]
    floor = (np.median(positive) if positive.size else 1.0) * 1e-3
    inv_scale = 1.0 / np.sqrt(np.maximum(tail_vals, floor))

    def tail_energy(features, predicted):
        residual = features - means[predicted]
        projected = residual @ tail_vecs
        projected *= inv_scale
        return np.sum(projected * projected, axis=1)

    pred_tr = np.argmax(z_tr, axis=1)
    pred_ev = np.argmax(z_ev, axis=1)
    train_energy = tail_energy(x, pred_tr)
    eval_energy = tail_energy(x_ev, pred_ev)

    sorted_by_pred = []
    fallback = np.sort(train_energy)
    for c in range(k):
        values = train_energy[pred_tr == c]
        sorted_by_pred.append(np.sort(values) if values.size >= 8 else fallback)

    out = np.empty(x_ev.shape[0], dtype=np.float64)
    for c in range(k):
        mask = pred_ev == c
        if np.any(mask):
            reference = sorted_by_pred[c]
            out[mask] = np.searchsorted(reference, eval_energy[mask], side="right") / reference.size

    scale = np.quantile(train_energy, 0.95) + 1e-12
    return out + 1e-4 * np.clip(eval_energy / scale, 0.0, 1e6)
