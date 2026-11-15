import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Mahalanobis density in bottleneck space can expose near-OOS intent shifts, while encoder-space density captures larger domain shifts; train-fitted scale normalization makes their additive combination comparable."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001", "m_0002"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = g_port.neg_log_density(tr.feats)
    full_train = g_full.neg_log_density(tr.pooled)

    port_med = np.median(port_train)
    full_med = np.median(full_train)
    port_scale = max(np.median(np.abs(port_train - port_med)), 1e-6)
    full_scale = max(np.median(np.abs(full_train - full_med)), 1e-6)

    port_score = (g_port.neg_log_density(ev.feats) - port_med) / port_scale
    full_score = (g_full.neg_log_density(ev.pooled) - full_med) / full_scale

    return 0.5 * port_score + 0.5 * full_score
