import numpy as np
from sklearn.utils.extmath import randomized_svd

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-style residual norm measures encoder variation outside the dominant in-scope feature subspace. Train-calibrated combination with energy emphasizes inputs that are both logit-uncertain and geometrically off-manifold."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0028"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    mean = x_train.mean(axis=0, keepdims=True)
    train_centered = x_train - mean
    eval_centered = x_eval - mean

    rank = min(192, train_centered.shape[1], max(1, train_centered.shape[0] - 1))
    _, _, vt = randomized_svd(
        train_centered,
        n_components=rank,
        n_iter=3,
        random_state=314159,
    )

    def residual_norm(x):
        total_sq = np.sum(x * x, axis=1)
        projected_sq = np.sum((x @ vt.T) ** 2, axis=1)
        return np.sqrt(np.maximum(total_sq - projected_sq, 0.0))

    train_residual = residual_norm(train_centered)
    eval_residual = residual_norm(eval_centered)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    def robust_scale(values):
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return median, max(1.4826 * mad, 1e-8)

    residual_center, residual_scale = robust_scale(train_residual)
    energy_center, energy_scale = robust_scale(train_energy)

    residual_z = (eval_residual - residual_center) / residual_scale
    energy_z = (eval_energy - energy_center) / energy_scale
    scores = residual_z + energy_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
