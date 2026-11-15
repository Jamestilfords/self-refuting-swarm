import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder densities capture near-intent and broad domain novelty respectively. Each density is converted to its training empirical lower-tail probability before additive fusion, preventing either representation's raw distance scale from dominating."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_port = g_port.log_density(tr.feats)
    train_full = g_full.log_density(tr.pooled)
    eval_port = g_port.log_density(ev.feats)
    eval_full = g_full.log_density(ev.pooled)

    port_normalizer = ctx.h.normalizer("ecdf", train_port)
    full_normalizer = ctx.h.normalizer("ecdf", train_full)

    port_tail = 1.0 - port_normalizer(eval_port)
    full_tail = 1.0 - full_normalizer(eval_full)

    port_scale = max(float(np.std(train_port)), 1e-6)
    full_scale = max(float(np.std(train_full)), 1e-6)
    w_port = 1.0 / port_scale
    w_full = 1.0 / full_scale
    total = w_port + w_full

    return (w_port * port_tail + w_full * full_tail) / total
