import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Energy is preserved exactly for confident predictions, while energy-uncertain queries are ranked by "
    "their residual distance outside a train-fitted shared within-intent tangent subspace. OOS inputs often "
    "contain bottleneck variation unsupported by ordinary within-intent directions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0183", "m_0188"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.feats, dtype=float)
    xev = np.asarray(ev.feats, dtype=float)
    ytr = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = xtr.shape[1]

    global_mean = np.mean(xtr, axis=0)
    means = np.tile(global_mean, (k, 1))
    counts = np.bincount(ytr, minlength=k)
    for c in range(k):
        if counts[c] > 0:
            means[c] = np.mean(xtr[ytr == c], axis=0)

    train_residuals = xtr - means[ytr]
    cov = (train_residuals.T @ train_residuals) / max(len(xtr), 1)
    vals, vecs = np.linalg.eigh(cov)
    rank = min(64, d)
    basis = vecs[:, -rank:]

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    def residual_energy(x, pred):
        z = x - means[pred]
        total = np.sum(z * z, axis=1)
        projected = np.sum((z @ basis) ** 2, axis=1)
        return np.maximum(total - projected, 0.0)

    rtr = residual_energy(xtr, train_pred)
    rev = residual_energy(xev, eval_pred)

    global_med = float(np.median(rtr))
    global_scale = float(np.median(np.abs(rtr - global_med))) + 1e-8
    med = np.full(k, global_med)
    scale = np.full(k, global_scale)

    for c in range(k):
        rc = rtr[train_pred == c]
        if len(rc) >= 8:
            mc = float(np.median(rc))
            sc = float(np.median(np.abs(rc - mc))) + 1e-8
            med[c] = mc
            scale[c] = sc

    energy_train = np.nan_to_num(
        ctx.h.energy(tr.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )
    energy_eval = np.nan_to_num(
        ctx.h.energy(ev.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )
    cutoff = float(np.quantile(energy_train, 0.70))

    z = (rev - med[eval_pred]) / scale[eval_pred]
    feature_rank = 0.5 + 0.5 * (z / (1.0 + np.abs(z)))

    scores = np.where(energy_eval <= cutoff, energy_eval, cutoff + 1.0 + feature_rank)
    return np.nan_to_num(scores, nan=cutoff, posinf=cutoff + 2.0, neginf=cutoff - 1.0)
