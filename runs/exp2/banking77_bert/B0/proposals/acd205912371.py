import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the score for low-uncertainty predictions, while only high-entropy predictions receive an additional penalty for weak Gaussian feature support. This targets queries whose classifier confidence is unreliable but whose representation is far from all known intents."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_entropy = ctx.h.entropy(tr.logits)
    eval_entropy = ctx.h.entropy(ev.logits)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    ent_cut = np.quantile(train_entropy, 0.60)
    ent_scale = np.quantile(train_entropy, 0.90) - ent_cut
    ent_scale = max(float(ent_scale), 1e-6)
    gate = np.maximum((eval_entropy - ent_cut) / ent_scale, 0.0)
    gate = np.minimum(gate, 2.0)

    energy_scale = np.std(train_energy)
    dist_scale = np.std(train_dist)
    energy_scale = max(float(energy_scale), 1e-6)
    dist_scale = max(float(dist_scale), 1e-6)

    distance_in_energy_units = (eval_dist - np.median(train_dist)) * (
        energy_scale / dist_scale
    )
    return eval_energy + 0.35 * gate * distance_in_energy_units
