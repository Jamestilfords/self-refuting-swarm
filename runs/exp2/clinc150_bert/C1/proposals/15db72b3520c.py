import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "MSP identifies classifier uncertainty while bottleneck Mahalanobis distance identifies representation-space novelty. Training-set empirical percentiles place both signals on a common calibrated scale, and their product emphasizes examples anomalous under both views."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_msp = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    eval_msp = np.asarray(ctx.h.msp(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    def empirical_percentile(train_values, values):
        reference = np.sort(np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        query = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ranks = np.searchsorted(reference, query, side="right")
        return ranks.astype(float) / float(max(len(reference), 1))

    msp_p = empirical_percentile(train_msp, eval_msp)
    dist_p = empirical_percentile(train_dist, eval_dist)

    scores = msp_p * dist_p + 0.15 * (msp_p + dist_p)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
