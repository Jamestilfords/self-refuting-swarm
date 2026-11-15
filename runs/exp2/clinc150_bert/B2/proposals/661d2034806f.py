import numpy as np
from scipy.special import ndtri

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-feature Mahalanobis distance capture complementary logit uncertainty and representation atypicality. Mapping each score to its train-fitted empirical normal scale prevents either raw score range from dominating the fusion."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0084"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)

    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def normal_tail_scale(train_values, values):
        train_values = np.sort(np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        n = max(train_values.size, 1)
        p = np.searchsorted(train_values, values, side="right") / (n + 1.0)
        p = np.clip(p, 1.0 / (n + 1.0), n / (n + 1.0))
        return ndtri(p)

    energy_z = normal_tail_scale(train_energy, eval_energy)
    distance_z = normal_tail_scale(train_distance, eval_distance)
    return (energy_z + distance_z) / np.sqrt(2.0)
