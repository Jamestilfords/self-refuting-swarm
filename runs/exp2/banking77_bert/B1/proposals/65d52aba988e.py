import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train empirical CDF calibration maps logit uncertainty and pooled-space class distance onto the same tail-probability scale. Taking their minimum emphasizes queries that are simultaneously classifier-uncertain and unsupported by the representation geometry."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_logit = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_logit = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def empirical_tail_scale(train_values, values):
        train_values = np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        )
        values = np.nan_to_num(
            values, nan=0.0, posinf=1e12, neginf=-1e12
        )
        reference = np.sort(train_values)
        return np.searchsorted(reference, values, side="right") / float(
            max(reference.size, 1)
        )

    logit_tail = empirical_tail_scale(train_logit, eval_logit)
    dist_tail = empirical_tail_scale(train_dist, eval_dist)

    joint_tail = np.minimum(logit_tail, dist_tail)
    tie_break = 1e-10 * np.tanh(
        np.nan_to_num(ev.logits[:, 0], nan=0.0, posinf=1e6, neginf=-1e6)
    )
    return np.asarray(joint_tail + tie_break, dtype=float)
