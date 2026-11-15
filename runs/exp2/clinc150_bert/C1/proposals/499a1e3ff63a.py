import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier uncertainty and bottleneck-space class distance are calibrated to their train-set empirical percentiles before a multiplicative fusion. The product emphasizes queries that are simultaneously atypical in representation space and poorly supported by the classifier."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)

    train_msp = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    eval_msp = np.asarray(ctx.h.msp(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    def empirical_scale(train_values, values):
        reference = np.sort(np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        query = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ranks = np.searchsorted(reference, query, side="right")
        return (ranks + 0.5) / (reference.size + 1.0)

    uncertainty = empirical_scale(train_msp, eval_msp)
    atypicality = empirical_scale(train_dist, eval_dist)

    return uncertainty * atypicality + 1e-6 * uncertainty
