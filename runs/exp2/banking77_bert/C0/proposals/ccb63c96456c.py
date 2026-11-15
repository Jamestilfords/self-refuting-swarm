import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Standardizing logit energy and pooled-feature Mahalanobis distance using training statistics puts complementary uncertainty signals on a comparable scale. Their sum flags inputs that are both logit-uncertain and distant from known intent representations."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    train_distance = g.min_mahalanobis_sq(tr.pooled)

    energy_mean = np.mean(train_energy)
    distance_mean = np.mean(train_distance)
    energy_scale = np.std(train_energy)
    distance_scale = np.std(train_distance)

    energy_scale = max(float(energy_scale), 1e-6)
    distance_scale = max(float(distance_scale), 1e-6)

    eval_energy = ctx.h.energy(ev.logits)
    eval_distance = g.min_mahalanobis_sq(ev.pooled)

    z_energy = (eval_energy - energy_mean) / energy_scale
    z_distance = (eval_distance - distance_mean) / distance_scale

    return np.nan_to_num(z_energy + z_distance, nan=0.0, posinf=1e6, neginf=-1e6)
