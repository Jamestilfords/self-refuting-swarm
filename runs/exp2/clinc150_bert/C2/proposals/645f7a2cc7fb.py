import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Queries with large encoder components outside the principal in-scope feature subspace are likely OOS. Combining this ViM-style residual with logit energy captures both representation novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = x_train.mean(axis=0)
    centered = x_train - mean
    cov = (centered.T @ centered) / max(len(x_train) - 1, 1)

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    total_var = float(vals.sum())
    if total_var > 1e-12:
        keep = int(np.searchsorted(np.cumsum(vals), 0.90 * total_var) + 1)
    else:
        keep = 1
    keep = min(max(keep, 1), vecs.shape[1])

    principal = vecs[:, :keep]
    eval_centered = x_eval - mean
    projected = eval_centered @ principal
    residual_sq = np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(projected * projected, axis=1),
        0.0,
    )
    residual = np.sqrt(residual_sq)

    residual_scale = np.sqrt(max(float(vals[keep:].sum()), 1e-12))
    residual_score = residual / residual_scale

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    energy_mean = float(train_energy.mean())
    energy_scale = float(train_energy.std())
    energy_scale = max(energy_scale, 1e-6)
    energy_score = (np.asarray(ctx.h.energy(ev.logits), dtype=np.float64) - energy_mean) / energy_scale

    scores = residual_score + energy_score
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
