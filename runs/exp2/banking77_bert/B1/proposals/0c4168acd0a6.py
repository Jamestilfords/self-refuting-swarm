import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to nearby unseen intents, while encoder-space Mahalanobis distance better captures domain-level semantic shifts; train-calibrated inverse-variance weighting puts both distances on comparable scales."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = g_port.min_mahalanobis_sq(tr.feats)
    full_train = g_full.min_mahalanobis_sq(tr.pooled)

    port_center = np.median(port_train)
    full_center = np.median(full_train)
    port_scale = np.median(np.abs(port_train - port_center)) * 1.4826
    full_scale = np.median(np.abs(full_train - full_center)) * 1.4826

    port_scale = max(float(port_scale), 1e-6)
    full_scale = max(float(full_scale), 1e-6)

    inv_port = 1.0 / port_scale
    inv_full = 1.0 / full_scale
    weight_sum = inv_port + inv_full
    w_port = inv_port / weight_sum
    w_full = inv_full / weight_sum

    port_eval = g_port.min_mahalanobis_sq(ev.feats)
    full_eval = g_full.min_mahalanobis_sq(ev.pooled)

    return w_port * ((port_eval - port_center) / port_scale) + w_full * (
        (full_eval - full_center) / full_scale
    )
