import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck density is sensitive to near-OOS examples that violate learned intent structure, "
    "while encoder density captures broader domain shifts. Train-fitted robust scale weights make "
    "their additive surprises comparable without using evaluation-set statistics."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0025"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_feat = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_pool = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_feat = g_feat.neg_log_density(tr.feats)
    train_pool = g_pool.neg_log_density(tr.pooled)
    eval_feat = g_feat.neg_log_density(ev.feats)
    eval_pool = g_pool.neg_log_density(ev.pooled)

    med_feat = np.median(train_feat)
    med_pool = np.median(train_pool)
    scale_feat = np.quantile(train_feat, 0.75) - np.quantile(train_feat, 0.25)
    scale_pool = np.quantile(train_pool, 0.75) - np.quantile(train_pool, 0.25)

    scale_feat = max(float(scale_feat), 1e-6)
    scale_pool = max(float(scale_pool), 1e-6)

    weight_feat = 1.0 / scale_feat
    weight_pool = 1.0 / scale_pool
    total_weight = weight_feat + weight_pool

    feat_surprise = (eval_feat - med_feat) / scale_feat
    pool_surprise = (eval_pool - med_pool) / scale_pool

    return (weight_feat * feat_surprise + weight_pool * pool_surprise) / total_weight
