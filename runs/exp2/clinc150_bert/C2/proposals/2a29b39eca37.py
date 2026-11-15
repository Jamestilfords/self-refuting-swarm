import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the dominant score for confident predictions, while predictive entropy gates in pooled-feature Mahalanobis distance when the classifier is uncertain. This emphasizes representation-space novelty primarily where logits provide ambiguous evidence."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    d_train = g.min_mahalanobis_sq(tr.pooled)
    d_eval = g.min_mahalanobis_sq(ev.pooled)

    e_train = ctx.h.energy(tr.logits)
    e_eval = ctx.h.energy(ev.logits)

    h_train = ctx.h.entropy(tr.logits)
    h_eval = ctx.h.entropy(ev.logits)

    ent_center = np.quantile(h_train, 0.70)
    ent_scale = np.quantile(h_train, 0.75) - np.quantile(h_train, 0.25)
    ent_scale = max(float(ent_scale), 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip((h_eval - ent_center) / ent_scale, -30.0, 30.0)))

    e_scale = np.quantile(e_train, 0.75) - np.quantile(e_train, 0.25)
    d_scale = np.quantile(d_train, 0.75) - np.quantile(d_train, 0.25)
    feature_weight = max(float(e_scale), 1e-6) / max(float(d_scale), 1e-6)

    return e_eval + gate * feature_weight * d_eval
