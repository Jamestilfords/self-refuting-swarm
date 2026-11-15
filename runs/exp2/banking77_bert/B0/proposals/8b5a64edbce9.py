import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and bottleneck-space class support capture complementary failures: uncertain logits and representations far from every known intent. Train-fitted robust scaling puts both signals on a shared scale before their joint OOS penalty is computed."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0096"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.feats)
    eval_dist = g.min_mahalanobis_sq(ev.feats)

    def robust_scale(x):
        q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
        scale = max(float(q3 - q1), 1e-6)
        return float(med), scale

    e_med, e_scale = robust_scale(train_energy)
    d_med, d_scale = robust_scale(train_dist)

    e_z = np.clip((eval_energy - e_med) / e_scale, -30.0, 30.0)
    d_z = np.clip((eval_dist - d_med) / d_scale, -30.0, 30.0)

    e_ood = 1.0 / (1.0 + np.exp(-e_z))
    d_ood = 1.0 / (1.0 + np.exp(-d_z))
    return -np.log(np.maximum((1.0 - e_ood) * (1.0 - d_ood), 1e-12))
