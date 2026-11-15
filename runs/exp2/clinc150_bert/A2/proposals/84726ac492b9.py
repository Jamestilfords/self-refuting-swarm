import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "ViM-style residual norms measure feature content outside the dominant in-scope representation subspace. "
    "Adding a train-calibrated residual anomaly to energy detects inputs that are confident yet geometrically unsupported."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0109"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)

    mean = np.mean(x_train, axis=0, keepdims=True)
    centered_train = x_train - mean

    # A moderately large principal subspace leaves a ViM-like null-space
    # residual while remaining inexpensive for the 256-dimensional bottleneck.
    _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    rank = min(128, vt.shape[0])
    basis = vt[:rank].T

    def residual_norm(x):
        z = x - mean
        projection = (z @ basis) @ basis.T
        return np.sqrt(np.sum((z - projection) ** 2, axis=1))

    train_residual = residual_norm(x_train)
    eval_residual = residual_norm(x_eval)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    energy_center = np.median(train_energy)
    residual_center = np.median(train_residual)
    energy_scale = np.median(np.abs(train_energy - energy_center))
    residual_scale = np.median(np.abs(train_residual - residual_center))

    if not np.isfinite(energy_scale) or energy_scale < 1e-12:
        energy_scale = np.std(train_energy) + 1e-12
    if not np.isfinite(residual_scale) or residual_scale < 1e-12:
        residual_scale = np.std(train_residual) + 1e-12

    scores = ((eval_energy - energy_center) / energy_scale +
              (eval_residual - residual_center) / residual_scale)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
