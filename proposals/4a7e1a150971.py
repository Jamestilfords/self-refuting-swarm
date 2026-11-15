import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is retained unchanged for training-calibrated confident queries, while only "
    "energy-uncertain queries receive a pooled-feature Mahalanobis support penalty."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0111", "m_0122"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    # The confidence boundary and transition width are fitted entirely on training data.
    boundary = float(np.quantile(train_energy, 0.65))
    width = float(np.quantile(train_energy, 0.85) - boundary)
    width = max(width, 1e-6)

    # Shared-covariance distance supplies representation support independently of logits.
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    # Calibrate only unusually large representation distances into a bounded penalty.
    d_center = float(np.quantile(train_dist, 0.75))
    d_scale = float(np.quantile(train_dist, 0.95) - d_center)
    d_scale = max(d_scale, 1e-6)
    penalty = np.maximum((eval_dist - d_center) / d_scale, 0.0)

    # Exactly preserve energy below the training confidence boundary.
    excess = np.maximum(eval_energy - boundary, 0.0)
    gate = 1.0 - np.exp(-excess / width)

    return np.asarray(eval_energy + gate * penalty, dtype=float)
