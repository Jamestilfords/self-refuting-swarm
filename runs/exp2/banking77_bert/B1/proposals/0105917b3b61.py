import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Train-calibrated robust scaling puts classifier energy and pooled-space "
    "Mahalanobis novelty on comparable units before combining their independent evidence."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0095", "m_0099"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = ctx.h.energy(tr.logits)
    train_distance = g.min_mahalanobis_sq(tr.pooled)

    def robust_scale(train_values, values):
        center = np.median(train_values)
        scale = np.median(np.abs(train_values - center)) * 1.4826
        scale = max(float(scale), 1e-6)
        return np.clip((values - center) / scale, -20.0, 20.0)

    energy_z = robust_scale(train_energy, ctx.h.energy(ev.logits))
    distance_z = robust_scale(train_distance, g.min_mahalanobis_sq(ev.pooled))

    joint = energy_z + distance_z + 0.25 * np.minimum(energy_z, distance_z)
    tie_break = 1e-10 * np.dot(ev.pooled, np.linspace(-1.0, 1.0, ev.pooled.shape[1]))
    return joint + tie_break
