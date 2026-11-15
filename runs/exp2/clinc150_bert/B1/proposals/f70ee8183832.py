import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck-space density is sensitive to near-OOS intent shifts, while encoder-output density captures broader domain shifts. Training-distribution interquantile ranges calibrate the two density scores before additive fusion."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = g_port.neg_log_density(tr.feats)
    full_train = g_full.neg_log_density(tr.pooled)

    port_scale = np.quantile(port_train, 0.90) - np.quantile(port_train, 0.10)
    full_scale = np.quantile(full_train, 0.90) - np.quantile(full_train, 0.10)

    port_scale = max(float(port_scale), 1e-8)
    full_scale = max(float(full_scale), 1e-8)

    w_port = 1.0 / port_scale
    w_full = 1.0 / full_scale
    weight_sum = w_port + w_full
    w_port /= weight_sum
    w_full /= weight_sum

    port_eval = g_port.neg_log_density(ev.feats)
    full_eval = g_full.neg_log_density(ev.pooled)

    scores = w_port * (port_eval / port_scale) + w_full * (full_eval / full_scale)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
