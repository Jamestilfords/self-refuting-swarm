import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Logit uncertainty and feature-space deviation are calibrated against their separate in-scope training distributions, then combined as a soft OR. This raises the OOS score when either classifier evidence or representation support is unusually anomalous without allowing one monotonically dominant component to determine the ranking."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_logit = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_logit = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def upper_ecdf(reference, values):
        reference = np.sort(np.nan_to_num(
            reference, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        return np.searchsorted(reference, values, side="right") / float(reference.size + 1)

    logit_tail = upper_ecdf(train_logit, eval_logit)
    distance_tail = upper_ecdf(train_dist, eval_dist)

    return 1.0 - (1.0 - logit_tail) * (1.0 - distance_tail)
