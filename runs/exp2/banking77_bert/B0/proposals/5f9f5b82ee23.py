import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder-space Mahalanobis densities capture complementary near- and far-OOS deviations. Training-only robust scale estimates place their distances on comparable units before additive fusion."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001", "m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_port = g_port.min_mahalanobis_sq(tr.feats)
    train_full = g_full.min_mahalanobis_sq(tr.pooled)
    eval_port = g_port.min_mahalanobis_sq(ev.feats)
    eval_full = g_full.min_mahalanobis_sq(ev.pooled)

    med_port = np.median(train_port)
    med_full = np.median(train_full)
    scale_port = np.percentile(train_port, 75) - np.percentile(train_port, 25)
    scale_full = np.percentile(train_full, 75) - np.percentile(train_full, 25)

    scale_port = max(float(scale_port), 1e-8)
    scale_full = max(float(scale_full), 1e-8)

    score_port = (eval_port - med_port) / scale_port
    score_full = (eval_full - med_full) / scale_full
    return np.nan_to_num(score_port + score_full, nan=0.0, posinf=1e12, neginf=-1e12)
