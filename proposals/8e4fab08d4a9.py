import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident queries retain the energy ordering, while energy-uncertain queries are "
    "ranked by how atypical their classifier-selected pooled-feature geometry is."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0047", "m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_class_d_all = g.class_mahalanobis_sq(tr.pooled)
    eval_class_d_all = g.class_mahalanobis_sq(ev.pooled)
    train_class_d = train_class_d_all[np.arange(len(tr.labels)), train_pred]
    eval_class_d = eval_class_d_all[np.arange(len(ev.logits)), eval_pred]

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    energy_ecdf = ctx.h.normalizer("ecdf", train_energy)
    distance_ecdf = ctx.h.normalizer("ecdf", train_class_d)

    e = np.clip(energy_ecdf(eval_energy), 0.0, 1.0)
    d = np.clip(distance_ecdf(eval_class_d), 0.0, 1.0)

    gate = 0.70
    scores = np.where(e <= gate, e, gate + (1.0 - gate) * d)
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
