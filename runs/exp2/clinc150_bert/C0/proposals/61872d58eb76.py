import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-feature Mahalanobis distance capture complementary uncertainty signals. Each is calibrated with train-set robust quantiles before fusion so neither raw scale dominates."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def calibrated_sigmoid(train_values, values):
        lo, mid, hi = np.quantile(train_values, [0.1, 0.5, 0.9])
        scale = max((hi - lo) / 2.0, 1e-6)
        z = np.clip((values - mid) / scale, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    energy_score = calibrated_sigmoid(train_energy, eval_energy)
    distance_score = calibrated_sigmoid(train_dist, eval_dist)
    return 0.5 * energy_score + 0.5 * distance_score
