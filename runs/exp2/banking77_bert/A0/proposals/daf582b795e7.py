import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Train-calibrated percentile scores place classifier uncertainty and bottleneck-space "
    "atypicality on a common scale. Their probabilistic-OR fusion flags queries that are "
    "suspicious by either signal while emphasizing agreement between the two."
)
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    train_msp = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    eval_msp = np.asarray(ctx.h.msp(ev.logits), dtype=float)

    def train_ecdf(train_values, values):
        train_values = np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ordered = np.sort(train_values)
        n = max(len(ordered), 1)
        return np.searchsorted(ordered, values, side="right") / float(n)

    p_logit = train_ecdf(train_msp, eval_msp)
    p_dist = train_ecdf(train_dist, eval_dist)

    scores = p_logit + p_dist - p_logit * p_dist
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
