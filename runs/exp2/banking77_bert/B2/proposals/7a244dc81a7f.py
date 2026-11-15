import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures weak or diffuse intent evidence, while pooled-space Mahalanobis distance captures representation-support violations. Train-calibrated empirical tail probabilities put both signals on a common bounded scale, and their interaction emphasizes examples jointly suspicious in logit and feature space."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0018"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_distance = g.min_mahalanobis_sq(tr.pooled)
    eval_distance = g.min_mahalanobis_sq(ev.pooled)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    def empirical_tail(train_values, values):
        reference = np.sort(np.asarray(train_values, dtype=float))
        n = reference.size
        if n == 0:
            return np.full(values.shape, 0.5, dtype=float)
        ranks = np.searchsorted(reference, values, side="right")
        return (ranks + 0.5) / (n + 1.0)

    energy_tail = empirical_tail(train_energy, eval_energy)
    distance_tail = empirical_tail(train_distance, eval_distance)

    joint_tail = energy_tail * distance_tail
    return 0.45 * energy_tail + 0.35 * distance_tail + 0.20 * joint_tail
