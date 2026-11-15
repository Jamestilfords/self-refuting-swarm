import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Confident classifier decisions retain the energy ranking, while examples in the train-calibrated high-uncertainty energy tail are ranked primarily by pooled-space Mahalanobis deviation. This conditional replacement prevents feature distance from perturbing easy in-scope decisions while making it decisive when logit support is weak."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0096", "m_0076"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_ref = np.sort(train_energy)
    d_ref = np.sort(train_dist)
    n_e = float(max(len(e_ref), 1))
    n_d = float(max(len(d_ref), 1))

    e_rank = np.searchsorted(e_ref, eval_energy, side="right") / n_e
    d_rank = np.searchsorted(d_ref, eval_dist, side="right") / n_d

    threshold = float(np.quantile(train_energy, 0.70))
    q25 = float(np.quantile(train_energy, 0.25))
    q75 = float(np.quantile(train_energy, 0.75))
    width = max(q75 - q25, 1e-6)

    z = np.clip(8.0 * (eval_energy - threshold) / width, -40.0, 40.0)
    uncertain = 1.0 / (1.0 + np.exp(-z))

    return (1.0 - uncertain) * e_rank + uncertain * d_rank
