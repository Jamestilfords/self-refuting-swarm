import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the primary ranking signal in the train-calibrated confident region, while a smooth train-fitted gate increasingly relies on pooled-feature Mahalanobis novelty for energy-uncertain queries. This targets confidently misclassified OOS items whose logits look plausible but whose representations are off the in-scope manifold."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0122"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = float(np.median(train_energy))
    e_scale = float(np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25))
    d_med = float(np.median(train_distance))
    d_scale = float(np.quantile(train_distance, 0.75) - np.quantile(train_distance, 0.25))
    e_scale = max(e_scale, 1e-6)
    d_scale = max(d_scale, 1e-6)

    e_train_z = (train_energy - e_med) / e_scale
    gate_center = float(np.quantile(e_train_z, 0.70))
    gate_width = max(
        float(np.quantile(e_train_z, 0.85) - gate_center),
        0.25,
    )

    e_z = (eval_energy - e_med) / e_scale
    d_z = (eval_distance - d_med) / d_scale
    gate_arg = np.clip((e_z - gate_center) / gate_width, -30.0, 30.0)
    gate = 1.0 / (1.0 + np.exp(-gate_arg))

    scores = (1.0 - gate) * e_z + gate * d_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
