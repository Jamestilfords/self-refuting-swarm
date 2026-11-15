import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the score for confidently in-scope-looking queries, while a train-calibrated full encoder-space Mahalanobis anomaly is activated only once energy enters its uncertain tail. This preserves energy's confident-item behavior and uses feature support to disambiguate uncertain queries."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0143"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    energy_sort = np.sort(train_energy)
    n_energy = max(len(energy_sort), 1)

    e_train = np.searchsorted(energy_sort, train_energy, side="right") / n_energy
    e_eval = np.searchsorted(energy_sort, eval_energy, side="right") / n_energy

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)
    dist_sort = np.sort(train_dist)
    n_dist = max(len(dist_sort), 1)
    d_eval = np.searchsorted(dist_sort, eval_dist, side="right") / n_dist

    uncertainty_start = float(np.quantile(e_train, 0.70))
    tail_width = max(float(np.quantile(e_train, 0.95) - uncertainty_start), 1e-6)
    gate = np.clip((e_eval - uncertainty_start) / tail_width, 0.0, 1.0)

    return e_eval + gate * d_eval
