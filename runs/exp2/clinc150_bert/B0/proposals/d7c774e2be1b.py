import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "High-margin predictions retain the energy ordering, while low-margin predictions "
    "handoff to pooled-feature Mahalanobis atypicality. This uses feature geometry only "
    "where the classifier's intent assignment is uncertain."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0099", "m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    def train_ecdf(train_values, values):
        ordered = np.sort(np.asarray(train_values, dtype=float))
        ranks = np.searchsorted(ordered, values, side="right")
        return ranks.astype(float) / float(max(len(ordered), 1))

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    energy_score = train_ecdf(train_energy, eval_energy)
    distance_score = train_ecdf(train_dist, eval_dist)

    train_margin = margin(tr.logits)
    eval_margin = margin(ev.logits)
    center = float(np.quantile(train_margin, 0.25))
    scale = float(np.quantile(train_margin, 0.75) - np.quantile(train_margin, 0.25))
    scale = max(scale, 1e-6)

    uncertain_gate = 1.0 / (1.0 + np.exp(np.clip((eval_margin - center) / scale, -40.0, 40.0)))
    return (1.0 - uncertain_gate) * energy_score + uncertain_gate * distance_score
