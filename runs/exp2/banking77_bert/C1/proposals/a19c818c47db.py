import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This gated fusion preserves the energy score for training-like confident predictions, while progressively adding pooled-feature Mahalanobis distance when classifier uncertainty is high. Ambiguous logits therefore rely more heavily on representation-space evidence of being out of distribution."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)
    train_u = ctx.h.msp(tr.logits)
    eval_u = ctx.h.msp(ev.logits)

    d_med = np.median(train_d)
    d_scale = np.median(np.abs(train_d - d_med)) * 1.4826 + 1e-8
    e_scale = np.median(np.abs(train_e - np.median(train_e))) * 1.4826 + 1e-8

    gate_center = np.quantile(train_u, 0.75)
    gate_scale = np.quantile(train_u, 0.90) - np.quantile(train_u, 0.25)
    gate_scale = max(float(gate_scale), 1e-8)

    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_u - gate_center) / gate_scale, -30.0, 30.0)))
    distance_term = e_scale * (eval_d - d_med) / d_scale

    return eval_e + gate * distance_term
