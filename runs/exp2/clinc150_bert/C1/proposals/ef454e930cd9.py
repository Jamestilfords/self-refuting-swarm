import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This preserves the energy ranking for confident predictions while adding shared-covariance feature distance only when classifier uncertainty is high. Uncertain OOS queries should therefore receive extra evidence from representation-space mismatch."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    train_conf = np.max(ctx.h.softmax(tr.logits), axis=1)
    eval_conf = np.max(ctx.h.softmax(ev.logits), axis=1)
    uncertainty = 1.0 - eval_conf

    train_uncertainty = 1.0 - train_conf
    threshold = np.quantile(train_uncertainty, 0.60)
    spread = np.quantile(train_uncertainty, 0.85) - threshold
    spread = max(float(spread), 1e-6)
    gate = np.clip((uncertainty - threshold) / spread, 0.0, 1.0)

    e_center = np.median(train_energy)
    e_scale = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)
    e_scale = max(float(e_scale), 1e-6)

    d_center = np.median(train_dist)
    d_scale = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    d_scale = max(float(d_scale), 1e-6)

    energy_z = (eval_energy - e_center) / e_scale
    distance_z = (eval_dist - d_center) / d_scale
    return energy_z + 0.5 * gate * distance_z
