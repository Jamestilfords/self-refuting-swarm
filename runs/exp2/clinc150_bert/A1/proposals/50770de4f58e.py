import numpy as np
from scipy.special import ndtri

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier energy and predicted-intent representation mismatch capture complementary OOS signals. Converting each to train-fitted normal scores makes their additive fusion insensitive to incompatible raw scales."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0119", "m_0101"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    n_classes = g.class_mahalanobis_sq(tr.pooled[:1]).shape[1]

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    tr_pred = np.clip(tr_pred, 0, n_classes - 1)
    ev_pred = np.clip(ev_pred, 0, n_classes - 1)

    tr_d_all = g.class_mahalanobis_sq(tr.pooled)
    ev_d_all = g.class_mahalanobis_sq(ev.pooled)
    tr_d = tr_d_all[np.arange(tr_d_all.shape[0]), tr_pred]
    ev_d = ev_d_all[np.arange(ev_d_all.shape[0]), ev_pred]

    tr_e = ctx.h.energy(tr.logits)
    ev_e = ctx.h.energy(ev.logits)

    def normal_score(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        ranks = np.searchsorted(ref, values, side="right")
        p = (ranks + 0.5) / (ref.size + 1.0)
        p = np.clip(p, 1e-5, 1.0 - 1e-5)
        return ndtri(p)

    energy_z = normal_score(tr_e, ev_e)
    distance_z = normal_score(tr_d, ev_d)
    return energy_z + distance_z
