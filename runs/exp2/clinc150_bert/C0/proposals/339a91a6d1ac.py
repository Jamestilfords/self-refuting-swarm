import numpy as np

FAMILY = "fusion"
ARM = "edl"
RATIONALE = "Robust train-calibrated sigmoid scales make vacuity and pooled-feature Mahalanobis distance comparable without using evaluation-set statistics. Their geometric combination emphasizes queries that are simultaneously epistemically uncertain and representation-space atypical."
SOURCE = ""
PARENTS = ["msp", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_logit = np.asarray(ctx.h.vacuity(tr.logits), dtype=float)
    eval_logit = np.asarray(ctx.h.vacuity(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def robust_sigmoid(train_values, values):
        train_values = np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        center = np.median(train_values)
        scale = 1.4826 * np.median(np.abs(train_values - center))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        z = np.clip((values - center) / scale, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    logit_tail = robust_sigmoid(train_logit, eval_logit)
    dist_tail = robust_sigmoid(train_dist, eval_dist)

    return np.asarray(
        np.sqrt(logit_tail * dist_tail) + 0.03 * logit_tail,
        dtype=float,
    )
