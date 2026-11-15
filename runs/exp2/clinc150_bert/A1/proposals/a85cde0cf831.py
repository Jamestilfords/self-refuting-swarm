import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck and pooled Mahalanobis densities capture complementary near-OOS and far-OOS shifts. "
    "A train-fitted covariance-aware additive weighting downweights redundant density signals without using evaluation statistics."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_scores = np.column_stack([
        g_port.min_mahalanobis_sq(tr.feats),
        g_full.min_mahalanobis_sq(tr.pooled),
    ])
    eval_scores = np.column_stack([
        g_port.min_mahalanobis_sq(ev.feats),
        g_full.min_mahalanobis_sq(ev.pooled),
    ])

    mean = np.mean(train_scores, axis=0)
    scale = np.std(train_scores, axis=0)
    scale = np.maximum(scale, 1e-8)

    z_train = (train_scores - mean) / scale
    z_eval = (eval_scores - mean) / scale

    cov = np.cov(z_train, rowvar=False)
    cov = np.atleast_2d(cov) + 0.05 * np.eye(2)
    precision = np.linalg.inv(cov)
    weights = precision @ np.ones(2)
    weights = np.maximum(weights, 1e-6)
    weights = weights / np.sum(weights)

    return z_eval @ weights
