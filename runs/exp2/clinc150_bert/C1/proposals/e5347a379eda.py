import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Confident in-scope examples retain their energy score, while examples with unusually uncertain energy are progressively replaced by a pooled-feature Mahalanobis distance. This lets representation distance resolve cases where logits alone are ambiguous."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_med = np.median(train_energy)
    e_q25, e_q75 = np.quantile(train_energy, [0.25, 0.75])
    e_scale = max(e_q75 - e_q25, 1e-6)

    d_med = np.median(train_dist)
    d_q25, d_q75 = np.quantile(train_dist, [0.25, 0.75])
    d_scale = max(d_q75 - d_q25, 1e-6)

    uncertainty_start = np.quantile(train_energy, 0.75)
    gate = 1.0 / (1.0 + np.exp(-np.clip(
        (eval_energy - uncertainty_start) / e_scale, -30.0, 30.0
    )))

    distance_on_energy_scale = e_med + e_scale * (eval_dist - d_med) / d_scale
    return (1.0 - gate) * eval_energy + gate * distance_on_energy_scale
