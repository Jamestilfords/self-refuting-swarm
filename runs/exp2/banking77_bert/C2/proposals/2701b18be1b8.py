import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This preserves the energy ordering for confident in-scope-like queries, while progressively adding class-conditional feature distance when energy indicates uncertainty. The gate is calibrated only on training energy values."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = np.median(train_energy)
    e_scale = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)
    e_scale = max(float(e_scale), 1e-6)

    d_med = np.median(train_dist)
    d_scale = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    d_scale = max(float(d_scale), 1e-6)

    uncertainty_center = np.quantile(train_energy, 0.70)
    gate_width = max(e_scale * 0.5, 1e-6)
    gate_arg = np.clip((eval_energy - uncertainty_center) / gate_width, -40.0, 40.0)
    gate = 1.0 / (1.0 + np.exp(-gate_arg))

    energy_z = (eval_energy - e_med) / e_scale
    distance_z = (eval_dist - d_med) / d_scale
    scores = energy_z + gate * distance_z

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
