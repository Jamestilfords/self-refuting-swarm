import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures feature energy outside the dominant training subspace, where atypical intents tend to lie. Training-calibrated fusion with energy combines representation mismatch with classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = x_train.mean(axis=0)
    centered = x_train - mean

    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)

    total = vals.sum()
    if total > 1e-12:
        descending = vals[::-1]
        keep = int(np.searchsorted(np.cumsum(descending), 0.90 * total) + 1)
        keep = max(1, min(keep, vecs.shape[1]))
        basis = vecs[:, -keep:]
    else:
        basis = vecs[:, -1:]

    train_norm2 = np.sum(centered * centered, axis=1)
    train_proj2 = np.sum((centered @ basis) ** 2, axis=1)
    train_resid = np.sqrt(np.maximum(train_norm2 - train_proj2, 0.0))

    eval_centered = x_eval - mean
    eval_norm2 = np.sum(eval_centered * eval_centered, axis=1)
    eval_proj2 = np.sum((eval_centered @ basis) ** 2, axis=1)
    eval_resid = np.sqrt(np.maximum(eval_norm2 - eval_proj2, 0.0))

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    resid_med = np.median(train_resid)
    resid_scale = np.median(np.abs(train_resid - resid_med)) + 1e-8
    energy_med = np.median(train_energy)
    energy_scale = np.median(np.abs(train_energy - energy_med)) + 1e-8

    residual_score = (eval_resid - resid_med) / resid_scale
    energy_score = (eval_energy - energy_med) / energy_scale
    return residual_score + energy_score
