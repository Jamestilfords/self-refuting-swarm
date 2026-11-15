import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier uncertainty and bottleneck-space support fail differently for OOS inputs. Train-fitted empirical tail maps put energy and Mahalanobis distance on a common scale, and a soft-union score increases when either signal is unusually OOS-like."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0167", "m_0010"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    train_distance = g.min_mahalanobis_sq(tr.feats)
    eval_distance = g.min_mahalanobis_sq(ev.feats)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    energy_tail = ctx.h.normalizer("ecdf", train_energy)
    distance_tail = ctx.h.normalizer("ecdf", train_distance)

    u_energy = np.clip(energy_tail(eval_energy), 0.0, 1.0 - 1e-12)
    u_distance = np.clip(distance_tail(eval_distance), 0.0, 1.0 - 1e-12)

    return -np.log((1.0 - u_energy) * (1.0 - u_distance))
