import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck density is sensitive to semantically nearby held-out intents, while encoder-output density better captures broad domain shifts. Training-calibrated anomaly scales give both representations comparable influence in the additive score."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    ld_port_train = g_port.log_density(tr.feats)
    ld_full_train = g_full.log_density(tr.pooled)

    n_port = ctx.h.normalizer("qsigmoid", ld_port_train)
    n_full = ctx.h.normalizer("qsigmoid", ld_full_train)

    a_port_train = 1.0 - n_port(ld_port_train)
    a_full_train = 1.0 - n_full(ld_full_train)

    s_port = max(float(np.std(a_port_train)), 1e-6)
    s_full = max(float(np.std(a_full_train)), 1e-6)
    w_port = 1.0 / s_port
    w_full = 1.0 / s_full
    weight_sum = w_port + w_full
    w_port /= weight_sum
    w_full /= weight_sum

    a_port_eval = 1.0 - n_port(g_port.log_density(ev.feats))
    a_full_eval = 1.0 - n_full(g_full.log_density(ev.pooled))

    return w_port * a_port_eval + w_full * a_full_eval
