import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "High-confidence predictions retain the energy ordering, while low maximum-softmax "
    "confidence smoothly shifts emphasis to pooled-feature Mahalanobis atypicality."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0072", "m_0084", "m_0093"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)
    tr_dist = g.min_mahalanobis_sq(tr.pooled)
    ev_dist = g.min_mahalanobis_sq(ev.pooled)

    energy_norm = ctx.h.normalizer("ecdf", tr_energy)
    dist_norm = ctx.h.normalizer("ecdf", tr_dist)

    tr_msp = np.max(ctx.h.softmax(tr.logits), axis=1)
    ev_msp = np.max(ctx.h.softmax(ev.logits), axis=1)
    uncertainty_train = 1.0 - tr_msp
    center = float(np.quantile(uncertainty_train, 0.50))
    spread = float(
        np.quantile(uncertainty_train, 0.80)
        - np.quantile(uncertainty_train, 0.20)
    )
    spread = max(spread, 1e-6)

    uncertainty_eval = 1.0 - ev_msp
    gate = 1.0 / (1.0 + np.exp(
        -4.0 * (uncertainty_eval - center) / spread
    ))

    scores = (1.0 - gate) * energy_norm(ev_energy) + gate * dist_norm(ev_dist)
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
