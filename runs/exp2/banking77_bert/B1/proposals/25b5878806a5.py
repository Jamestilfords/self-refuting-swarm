import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is retained for confident in-scope-like predictions, while a train-calibrated "
    "gate increasingly substitutes bottleneck Mahalanobis distance when classifier energy is uncertain."
)
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0017", "m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    e_center = np.median(train_energy)
    d_center = np.median(train_dist)
    e_scale = np.median(np.abs(train_energy - e_center)) * 1.4826
    d_scale = np.median(np.abs(train_dist - d_center)) * 1.4826
    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)

    train_ez = (train_energy - e_center) / e_scale
    eval_ez = (eval_energy - e_center) / e_scale
    eval_dz = (eval_dist - d_center) / d_scale

    gate_center = float(np.quantile(train_ez, 0.70))
    gate_scale = max(float(np.std(train_ez)), 0.5)
    gate_arg = np.clip((eval_ez - gate_center) / gate_scale, -30.0, 30.0)
    gate = 1.0 / (1.0 + np.exp(-gate_arg))

    return (1.0 - gate) * eval_ez + gate * eval_dz
