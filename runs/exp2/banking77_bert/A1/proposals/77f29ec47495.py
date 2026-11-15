import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the dominant signal for confidently in-scope queries, while a train-calibrated encoder Mahalanobis distance is increasingly emphasized as energy indicates classifier uncertainty. This conditional fusion avoids applying representation novelty equally to already confident predictions."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    train_distance = g.min_mahalanobis_sq(tr.pooled)

    eval_energy = ctx.h.energy(ev.logits)
    eval_distance = g.min_mahalanobis_sq(ev.pooled)

    energy_sorted = np.sort(train_energy)
    distance_sorted = np.sort(train_distance)

    def train_cdf(values, sorted_train):
        ranks = np.searchsorted(sorted_train, values, side="right")
        return ranks.astype(float) / float(len(sorted_train) + 1)

    energy_oos = train_cdf(eval_energy, energy_sorted)
    distance_oos = train_cdf(eval_distance, distance_sorted)

    uncertainty_gate = energy_oos * energy_oos
    return (1.0 - uncertainty_gate) * energy_oos + uncertainty_gate * distance_oos
