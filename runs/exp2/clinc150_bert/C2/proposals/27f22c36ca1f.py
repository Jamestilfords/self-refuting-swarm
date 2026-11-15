import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Gaussian density is sensitive to near-OOS held-out intents, while encoder-space density captures broader domain shifts. Train-fitted inverse-dispersion weights put both negative log-density scores on comparable scales without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = -g_port.log_density(tr.feats)
    full_train = -g_full.log_density(tr.pooled)

    port_med = np.median(port_train)
    full_med = np.median(full_train)
    port_scale = np.percentile(port_train, 75) - np.percentile(port_train, 25)
    full_scale = np.percentile(full_train, 75) - np.percentile(full_train, 25)

    port_scale = max(float(port_scale), 1e-6)
    full_scale = max(float(full_scale), 1e-6)

    inv_port = 1.0 / port_scale
    inv_full = 1.0 / full_scale
    total = inv_port + inv_full
    w_port = inv_port / total
    w_full = inv_full / total

    port_eval = -g_port.log_density(ev.feats)
    full_eval = -g_full.log_density(ev.pooled)

    scores = w_port * (port_eval - port_med) / port_scale
    scores += w_full * (full_eval - full_med) / full_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
