import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "In-scope encoder representations should lie near a principal subspace learned from training data, while OOS inputs often retain a larger residual component. Combining this residual with energy captures both representation-space novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = np.mean(x_train, axis=0)
    centered_train = x_train - mean
    centered_eval = x_eval - mean

    cov = centered_train.T @ centered_train
    cov /= max(x_train.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)

    rank = min(128, x_train.shape[1], max(x_train.shape[0] - 1, 1))
    basis = vecs[:, -rank:]

    train_proj = centered_train @ basis
    eval_proj = centered_eval @ basis
    train_residual = np.sqrt(np.maximum(
        np.sum(centered_train * centered_train, axis=1)
        - np.sum(train_proj * train_proj, axis=1),
        0.0,
    ))
    eval_residual = np.sqrt(np.maximum(
        np.sum(centered_eval * centered_eval, axis=1)
        - np.sum(eval_proj * eval_proj, axis=1),
        0.0,
    ))

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    residual_med = np.median(train_residual)
    residual_scale = np.median(np.abs(train_residual - residual_med)) + 1e-8
    energy_med = np.median(train_energy)
    energy_scale = np.median(np.abs(train_energy - energy_med)) + 1e-8

    residual_z = (eval_residual - residual_med) / residual_scale
    energy_z = (eval_energy - energy_med) / energy_scale
    return residual_z + energy_z
