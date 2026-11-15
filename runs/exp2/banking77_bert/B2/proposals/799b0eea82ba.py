import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Logit energy captures classifier uncertainty, while pooled-space Mahalanobis distance captures departures from class-conditional representation support. Robust train-only scaling lets both complementary signals contribute on a comparable scale."
SOURCE = "https://www.emergentmind.com/papers/2101.03778"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_distance = g.min_mahalanobis_sq(tr.pooled)
    eval_distance = g.min_mahalanobis_sq(ev.pooled)

    def robust_scale(train_values, values):
        median = np.median(train_values)
        q25, q75 = np.percentile(train_values, [25.0, 75.0])
        scale = q75 - q25
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        return (values - median) / scale

    energy_z = robust_scale(train_energy, eval_energy)
    distance_z = robust_scale(train_distance, eval_distance)
    return energy_z + distance_z
