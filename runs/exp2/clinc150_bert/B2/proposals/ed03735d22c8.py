import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Low-energy items retain the energy ranking, while items in the train-calibrated "
    "high-energy uncertainty region are increasingly ranked by pooled-space class support."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0041"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_mean = float(np.mean(train_energy))
    e_std = float(np.std(train_energy))
    d_mean = float(np.mean(train_dist))
    d_std = float(np.std(train_dist))
    e_std = max(e_std, 1e-8)
    d_std = max(d_std, 1e-8)

    e_train_z = (train_energy - e_mean) / e_std
    e_eval_z = (eval_energy - e_mean) / e_std
    d_eval_z = (eval_dist - d_mean) / d_std

    center = float(np.quantile(e_train_z, 0.75))
    width = float(np.quantile(e_train_z, 0.90) - np.quantile(e_train_z, 0.50))
    width = max(width, 0.25)

    gate = 1.0 / (1.0 + np.exp(-np.clip((e_eval_z - center) / width, -30.0, 30.0)))
    return (1.0 - gate) * e_eval_z + gate * d_eval_z
