import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Low-energy confident queries retain the classifier's energy ordering, while high-energy uncertain queries are evaluated primarily by their pooled-feature distance from in-scope classes. This conditional gate uses representation novelty only where logits indicate that the classifier is unreliable."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = gda.min_mahalanobis_sq(tr.pooled)
    eval_dist = gda.min_mahalanobis_sq(ev.pooled)

    energy_cdf = ctx.h.normalizer("ecdf", train_energy)
    dist_cdf = ctx.h.normalizer("ecdf", train_dist)

    e_train = energy_cdf(train_energy)
    e_eval = energy_cdf(eval_energy)
    d_eval = dist_cdf(eval_dist)

    gate_center = np.quantile(e_train, 0.70)
    gate_scale = np.quantile(e_train, 0.85) - np.quantile(e_train, 0.50)
    gate_scale = max(float(gate_scale), 0.05)

    gate = 1.0 / (1.0 + np.exp(-(e_eval - gate_center) / gate_scale))
    scores = (1.0 - gate) * e_eval + gate * d_eval
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
