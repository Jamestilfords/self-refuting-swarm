import numpy as np
from scipy.special import expit

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This conditional fusion preserves energy-based ranking for queries that look confident under the classifier, but switches toward encoder-space class distance once predictive uncertainty exceeds a train-calibrated threshold."
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

    train_u = ctx.h.entropy(tr.logits)
    eval_u = ctx.h.entropy(ev.logits)

    def ecdf_reference(reference, values):
        reference = np.sort(np.nan_to_num(
            reference, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        return np.searchsorted(reference, values, side="right") / float(len(reference))

    energy_pct = ecdf_reference(train_e, eval_e)
    distance_pct = ecdf_reference(train_d, eval_d)

    center = float(np.percentile(train_u, 70))
    spread = float(
        np.percentile(train_u, 75) - np.percentile(train_u, 25)
    )
    spread = max(spread, 1e-6)

    uncertainty_gate = expit(4.0 * (eval_u - center) / spread)

    # At low uncertainty this is nearly the energy percentile; at high
    # uncertainty it becomes the feature-distance percentile.
    return (1.0 - uncertainty_gate) * energy_pct + uncertainty_gate * distance_pct
