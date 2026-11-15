import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is retained for confidently in-scope-like queries, while a train-calibrated "
    "local encoder-neighborhood distance is activated mainly for energy-uncertain queries."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    e_med = float(np.median(train_energy))
    e_iqr = float(np.percentile(train_energy, 75) - np.percentile(train_energy, 25))
    e_scale = max(e_iqr, 1e-6)

    # With training points as queries, the nearest item is itself; k=21 therefore
    # approximates the radius to the twentieth distinct local neighbor.
    train_radius = ctx.h.knn(tr.pooled, tr.pooled, k=21, l2=True)
    eval_radius = ctx.h.knn(tr.pooled, ev.pooled, k=20, l2=True)

    r_med = float(np.median(train_radius))
    r_iqr = float(np.percentile(train_radius, 75) - np.percentile(train_radius, 25))
    r_scale = max(r_iqr, 1e-6)

    radius_tail = (eval_radius - r_med) / r_scale
    uncertainty = (eval_energy - e_med) / e_scale
    gate = 1.0 / (1.0 + np.exp(-np.clip(uncertainty, -30.0, 30.0)))

    return eval_energy + gate * radius_tail
