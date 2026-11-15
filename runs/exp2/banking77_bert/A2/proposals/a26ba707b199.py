import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures classifier uncertainty while full encoder-space Mahalanobis distance measures departure from class-conditional feature support. Train-calibrated ECDF scaling places both signals on a comparable bounded scale without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0001", "m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    train_distance = g.min_mahalanobis_sq(tr.pooled)

    energy_norm = ctx.h.normalizer("ecdf", train_energy)
    distance_norm = ctx.h.normalizer("ecdf", train_distance)

    e = energy_norm(ctx.h.energy(ev.logits))
    d = distance_norm(g.min_mahalanobis_sq(ev.pooled))
    return np.nan_to_num(e + d, nan=0.0, posinf=2.0, neginf=0.0)
