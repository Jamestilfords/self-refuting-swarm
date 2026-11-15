import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and full encoder-space Mahalanobis distance capture complementary classifier uncertainty and representation atypicality. Empirical train-distribution calibration puts both signals on a common robust scale without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_distance = g.min_mahalanobis_sq(tr.pooled)
    eval_distance = g.min_mahalanobis_sq(ev.pooled)

    def calibrated_rank(train_values, values):
        ordered = np.sort(np.asarray(train_values, dtype=float))
        ranks = np.searchsorted(ordered, np.asarray(values, dtype=float), side="right")
        p = ranks / float(ordered.size + 1)
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        return np.log(p) - np.log1p(-p)

    energy_score = calibrated_rank(train_energy, eval_energy)
    distance_score = calibrated_rank(train_distance, eval_distance)
    return energy_score + distance_score
