import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Train-fitted robust scaling places energy uncertainty and pooled-space Mahalanobis novelty "
    "on a common scale, while their multiplicative agreement emphasizes queries abnormal in both views."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0072", "m_0076"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def robust_z(reference, values):
        reference = np.nan_to_num(
            reference, nan=0.0, posinf=1e12, neginf=-1e12
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        center = np.median(reference)
        scale = 1.4826 * np.median(np.abs(reference - center))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(reference) + 1e-8
        return np.clip((values - center) / scale, -40.0, 40.0)

    z_energy = robust_z(train_energy, eval_energy)
    z_dist = robust_z(train_dist, eval_dist)

    p_energy = 1.0 / (1.0 + np.exp(-z_energy))
    p_dist = 1.0 / (1.0 + np.exp(-z_dist))

    scores = p_energy * p_dist + 1e-4 * (z_energy + z_dist)
    return np.nan_to_num(scores, nan=0.0, posinf=100.0, neginf=-100.0)
