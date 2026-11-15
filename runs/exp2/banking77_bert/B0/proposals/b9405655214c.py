import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-calibrated empirical tail probabilities put classifier uncertainty and bottleneck Mahalanobis novelty on a common evidence scale. Adding clipped log-odds emphasizes queries that are simultaneously atypical in logits and representation space."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0018", "m_0022", "m_0034"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    tr_msp = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    ev_msp = np.asarray(ctx.h.msp(ev.logits), dtype=float)
    tr_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    ev_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    def upper_tail(train_values, values):
        ref = np.sort(np.nan_to_num(train_values, nan=0.0, posinf=1e12, neginf=-1e12))
        val = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ranks = np.searchsorted(ref, val, side="right")
        p = (ranks + 0.5) / (ref.size + 1.0)
        return np.clip(p, 1e-4, 1.0 - 1e-4)

    p_msp = upper_tail(tr_msp, ev_msp)
    p_dist = upper_tail(tr_dist, ev_dist)

    evidence_msp = np.log(p_msp) - np.log1p(-p_msp)
    evidence_dist = np.log(p_dist) - np.log1p(-p_dist)
    return evidence_msp + evidence_dist
